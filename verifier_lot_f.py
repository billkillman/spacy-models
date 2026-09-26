"""NUM-123 — Vérifier que le lot F est démonstratif, avant tout envoi à Arize.

Le lot F doit déclencher le moniteur d'embeddings **seul**. Ce script le vérifie hors ligne,
avec les vrais services et les seuils déjà calculés, en comparant F à son témoin apparié E
(les mêmes réclamations, sans le contexte ajouté) :

    signal                        témoin E           lot F                attendu pour F
    dérive d'embeddings           sous le seuil      au-dessus du seuil   alerte
    répartition par familles      sous le seuil      sous le seuil        pas d'alerte
    part hors périmètre           sous le seuil      sous le seuil        pas d'alerte

Il mesure aussi, réclamation par réclamation, combien changent de famille entre E et F, et
**quel contexte** en est responsable : c'est ce diagnostic qui permet d'ajuster le lot.

Coût : les embeddings de F en 4 requêtes ; le reranker en une requête par réclamation de F,
espacées de 6,5 s (une dizaine de minutes pour 100). Rien n'est appelé pour E : ses
embeddings et ses motifs figurent déjà dans les références de NUM-109 et NUM-122. Les
résultats sont mémorisés : une relance reprend là où elle s'était arrêtée.

Usage
-----
    python verifier_lot_f.py --lots lots --reference reference_signaux \\
        --reference-embeddings reference_embeddings

Variables d'environnement (les mêmes que l'API) : LLM_API_URL, LLM_API_KEY, CA_BUNDLE ;
RERANK_MODEL, RERANK_PATH, EMBEDDING_PATH si différents des défauts.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

import signaux_entree as sig
from calculer_reference import calculer_signaux, proportions, psi
from calculer_reference_embeddings import embeddings_avec_cache, matrice_normalisee

#: Marge souhaitée sur la dérive d'embeddings : Arize calcule sa propre distance, qui peut
#: différer un peu de la nôtre. Sous cette marge, le déclenchement réel n'est pas assuré.
MARGE_RECOMMANDEE = 1.2
#: Au-delà de cette part de réclamations changeant de famille, un contexte est jugé « fuyant ».
SEUIL_CONTEXTE_FUYANT = 0.15
#: Famille attribuée à une réclamation hors périmètre, lue dans la table des familles.
FAMILLE_HORS_PERIMETRE = sig.famille_motif(sig.HORS_PERIMETRE)


# =============================================================================
# Lecture des références
# =============================================================================

def lire_json(chemin: Path) -> dict | list:
    if not chemin.exists():
        raise SystemExit(f"Fichier introuvable : {chemin}")
    return json.loads(chemin.read_text(encoding="utf-8"))


def lire_reference_signaux(dossier: Path) -> tuple[dict, dict]:
    """Motif, famille et couverture de chaque réclamation du pilote, et seuils de NUM-109."""
    with (dossier / "reference_signaux.csv").open(encoding="utf-8", newline="") as flux:
        lignes = {l["claim_id"]: l for l in csv.DictReader(flux)}
    if not lignes or "famille" not in next(iter(lignes.values())):
        raise SystemExit("reference_signaux.csv sans colonne « famille » : relancer NUM-109 avec le reranker.")
    return lignes, lire_json(dossier / "seuils_signaux.json")


def lire_reference_embeddings(dossier: Path) -> tuple[dict, dict]:
    vecteurs = {}
    for ligne in (dossier / "vecteurs_reference.jsonl").read_text(encoding="utf-8").splitlines():
        if ligne.strip():
            entree = json.loads(ligne)
            vecteurs[entree["claim_id"]] = entree["vecteur"]
    return vecteurs, lire_json(dossier / "seuils_embeddings.json")


def seuil(manifeste: dict, nom: str) -> float:
    bloc = manifeste["seuils_moniteurs"][nom]
    return float(next(v for k, v in bloc.items() if k.startswith("seuil_p")))


# =============================================================================
# Mesures
# =============================================================================

def distance_au_centroide(vecteurs: np.ndarray, centre: np.ndarray) -> float:
    return float(np.linalg.norm(vecteurs.mean(axis=0) - centre))


def mesurer_familles(familles: list[str], reference: dict) -> dict:
    return {"psi_familles": round(psi(reference, proportions(familles)), 4),
            "part_hors_perimetre": round(familles.count(FAMILLE_HORS_PERIMETRE) / len(familles), 4)}


# =============================================================================
# Pilotage
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="NUM-123 — le lot F est-il démonstratif ?")
    parser.add_argument("--lots", default="lots", help="dossier produit par generer_lots.py")
    parser.add_argument("--reference", default="reference_signaux", help="dossier produit en NUM-109")
    parser.add_argument("--reference-embeddings", default="reference_embeddings", help="dossier produit en NUM-122")
    parser.add_argument("--modeles", default="data/business_response_models.json")
    parser.add_argument("--pause", type=float, default=6.5, help="secondes entre deux appels au reranker")
    parser.add_argument("--intervalle", type=float, default=6.5,
                        help="secondes minimum entre deux requêtes au modèle d'embedding")
    parser.add_argument("--sortie", default="verification_lot_f")
    args = parser.parse_args()
    logging.basicConfig(level=logging.WARNING, format="    %(message)s")
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    lots = Path(args.lots)
    lot_e = lire_json(lots / "lot_E_controle.json")
    lot_f = lire_json(lots / "lot_F_glissement_semantique.json")
    ref_signaux, seuils_signaux = lire_reference_signaux(Path(args.reference))
    ref_vecteurs, seuils_emb = lire_reference_embeddings(Path(args.reference_embeddings))
    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)

    tailles = {seuils_signaux["parametres"]["taille_fenetre"], seuils_emb["seuil_derive"]["taille_fenetre"]}
    if tailles != {len(lot_f)}:
        print(f"\n  Attention : seuils calibrés pour des fenêtres de {sorted(tailles)}, lot F de {len(lot_f)}.")
    print(f"\n  Lot F : {len(lot_f)} réclamations, intensité {lot_f[0]['intensite']} — témoin : lot E")

    # ------------------------------------------------------------ embeddings
    modele = seuils_emb["modele"]
    embedder = sig.Embedder(url=os.environ["LLM_API_URL"], cle=os.environ["LLM_API_KEY"], modele=modele["nom"],
                            chemin=os.getenv("EMBEDDING_PATH", modele.get("chemin", "/embeddings")),
                            verification=os.getenv("CA_BUNDLE") or True, intervalle_min_s=args.intervalle)
    print(f"\n  Embeddings ({modele['nom']})…")
    empreinte = sig.empreinte_modele(embedder)
    if empreinte != modele["empreinte"]:
        print(f"    Empreinte du modèle changée ({empreinte} au lieu de {modele['empreinte']}) : le service a "
              "changé de modèle, les vecteurs ne sont plus comparables à la référence. Refaire NUM-122.")
        return 2

    ids_ref = sorted(ref_vecteurs)
    centre = np.array([ref_vecteurs[i] for i in ids_ref]).mean(axis=0)
    manquants = [e["source_id"] for e in lot_e if e["source_id"] not in ref_vecteurs]
    if manquants:
        raise SystemExit(f"Réclamations du lot E absentes de la référence d'embeddings : {manquants[:5]}")
    vecteurs_e = np.array([ref_vecteurs[e["source_id"]] for e in lot_e])
    lignes_f = [{"claim_id": f["claimId"], "texte": f["claimVerbatim"]} for f in lot_f]
    bruts_f = embeddings_avec_cache(embedder, lignes_f, sortie / "cache_embeddings_lot_f.jsonl")
    vecteurs_f = matrice_normalisee(bruts_f, [l["claim_id"] for l in lignes_f])

    seuil_emb = float(seuils_emb["seuil_derive"]["distance_euclidienne_des_centroides"])
    dist_e, dist_f = distance_au_centroide(vecteurs_e, centre), distance_au_centroide(vecteurs_f, centre)
    source_e = {e["claimId"]: i for i, e in enumerate(lot_e)}
    similarite_appariee = float(np.mean([vecteurs_f[i] @ vecteurs_e[source_e[f["source_lot_e"]]]
                                         for i, f in enumerate(lot_f)]))

    # ------------------------------------------------------------ reranker
    documents = sig.documents_modeles(Path(args.modeles))
    reranker = sig.Reranker(url=os.environ["LLM_API_URL"], cle=os.environ["LLM_API_KEY"],
                            modele=os.getenv("RERANK_MODEL", seuils_signaux["reranker"]["modele"]),
                            chemin=os.getenv("RERANK_PATH", seuils_signaux["reranker"]["chemin"]),
                            verification=os.getenv("CA_BUNDLE") or True)
    seuil_hp = float(seuils_signaux["seuil_hors_perimetre"]["seuil"])
    print(f"\n  Reranker ({reranker.modele}) sur le lot F…")
    calculer_signaux(lignes_f, reranker, documents, sortie / "cache_reranker_lot_f.jsonl", args.pause)

    familles_ref = proportions([l["famille"] for l in ref_signaux.values()])
    familles_e = [ref_signaux[e["source_id"]]["famille"] for e in lot_e]
    familles_f = []
    for l in lignes_f:
        motif = sig.HORS_PERIMETRE if l["score_couverture"] < seuil_hp else l["motif_premier"]
        familles_f.append(sig.famille_motif(motif))

    mesure_e, mesure_f = mesurer_familles(familles_e, familles_ref), mesurer_familles(familles_f, familles_ref)
    seuil_psi, seuil_part = seuil(seuils_signaux, "psi_familles"), seuil(seuils_signaux, "part_hors_perimetre")

    # ------------------------------------------------------------ diagnostic apparié
    changements, par_contexte = Counter(), defaultdict(lambda: [0, 0])
    for f, fam_e, fam_f in zip(lot_f, [familles_e[source_e[f["source_lot_e"]]] for f in lot_f], familles_f):
        par_contexte[f["contexte"]][1] += 1
        if fam_e != fam_f:
            changements[(fam_e, fam_f)] += 1
            par_contexte[f["contexte"]][0] += 1
    fuyants = {c: round(n / t, 3) for c, (n, t) in par_contexte.items() if t and n / t > SEUIL_CONTEXTE_FUYANT}
    longueur_e = np.median([sig.descripteurs(e["claimVerbatim"])["longueur_mots"] for e in lot_e])
    longueur_f = np.median([l["longueur_mots"] for l in lignes_f])

    # ------------------------------------------------------------ verdict
    criteres = {
        "embeddings_F_au_dessus_du_seuil": dist_f > seuil_emb,
        "embeddings_E_sous_le_seuil": dist_e <= seuil_emb,
        "familles_F_sous_le_seuil": mesure_f["psi_familles"] <= seuil_psi,
        "hors_perimetre_F_sous_le_seuil": mesure_f["part_hors_perimetre"] <= seuil_part,
    }
    marge = dist_f / seuil_emb if seuil_emb else float("inf")
    demonstratif = all(criteres.values())
    conseils = []
    if not criteres["embeddings_F_au_dessus_du_seuil"] or (demonstratif and marge < MARGE_RECOMMANDEE):
        conseils.append("Décalage trop faible : régénérer le lot avec --intensite 2.")
    if not (criteres["familles_F_sous_le_seuil"] and criteres["hors_perimetre_F_sous_le_seuil"]):
        conseils.append("Le contexte déplace les familles : " + (
            f"retirer ou reformuler les contextes fuyants {sorted(fuyants)}" if fuyants
            else "revenir à --intensite 1") + ".")
    if not criteres["embeddings_E_sous_le_seuil"]:
        conseils.append("Le témoin E dépasse déjà le seuil : le seuil de NUM-122 est trop serré, le revoir.")

    rapport = {
        "genere_le": datetime.now().isoformat(timespec="seconds"),
        "lot_f": {"reclamations": len(lot_f), "intensite": lot_f[0]["intensite"]},
        "embeddings": {"modele": modele["nom"], "empreinte_verifiee": empreinte, "seuil": seuil_emb,
                       "distance_temoin_E": round(dist_e, 5), "distance_lot_F": round(dist_f, 5),
                       "marge_F_sur_seuil": round(marge, 3),
                       "similarite_moyenne_F_E_appariee": round(similarite_appariee, 4)},
        "familles": {"seuil_psi": seuil_psi, "seuil_part_hors_perimetre": seuil_part,
                     "temoin_E": mesure_e, "lot_F": mesure_f,
                     "reclamations_changeant_de_famille": sum(changements.values()),
                     "transitions": {f"{a} -> {b}": n for (a, b), n in changements.most_common()},
                     "contextes_fuyants": fuyants},
        "surface": {"longueur_mediane_E": float(longueur_e), "longueur_mediane_F": float(longueur_f),
                    "note": "hausse attendue ; la longueur est affichée, non alertante (NUM-115)"},
        "criteres": criteres, "demonstratif": demonstratif, "conseils": conseils,
    }
    (sortie / "rapport_lot_f.json").write_text(json.dumps(rapport, ensure_ascii=False, indent=2), encoding="utf-8")
    afficher(rapport)
    return 0 if demonstratif else 1


def afficher(r: dict) -> None:
    e, f = r["embeddings"], r["familles"]
    ok = lambda b: "oui" if b else "NON"
    print(f"\n  {'signal':<28} {'témoin E':>10} {'lot F':>10} {'seuil':>8}   attendu pour F")
    print(f"  {'dérive d embeddings':<28} {e['distance_temoin_E']:>10.4f} {e['distance_lot_F']:>10.4f} "
          f"{e['seuil']:>8.4f}   au-dessus : {ok(r['criteres']['embeddings_F_au_dessus_du_seuil'])}")
    print(f"  {'répartition par familles':<28} {f['temoin_E']['psi_familles']:>10.4f} "
          f"{f['lot_F']['psi_familles']:>10.4f} {f['seuil_psi']:>8.4f}   sous le seuil : "
          f"{ok(r['criteres']['familles_F_sous_le_seuil'])}")
    print(f"  {'part hors périmètre':<28} {f['temoin_E']['part_hors_perimetre']:>10.4f} "
          f"{f['lot_F']['part_hors_perimetre']:>10.4f} {f['seuil_part_hors_perimetre']:>8.4f}   sous le seuil : "
          f"{ok(r['criteres']['hors_perimetre_F_sous_le_seuil'])}")
    print(f"\n  Marge de F sur le seuil d'embeddings : ×{e['marge_F_sur_seuil']:.2f} "
          f"(recommandé ≥ ×{MARGE_RECOMMANDEE})")
    print(f"  Similarité moyenne F / E, réclamation par réclamation : {e['similarite_moyenne_F_E_appariee']:.3f}")
    print(f"  Réclamations changeant de famille entre E et F : {f['reclamations_changeant_de_famille']}")
    for transition, n in list(f["transitions"].items())[:5]:
        print(f"    {n:>3}  {transition}")
    if f["contextes_fuyants"]:
        print(f"  Contextes fuyants (> {SEUIL_CONTEXTE_FUYANT:.0%} de changements) : {f['contextes_fuyants']}")
    print(f"\n  Lot F {'DÉMONSTRATIF' if r['demonstratif'] else 'NON DÉMONSTRATIF'}")
    for c in r["conseils"]:
        print(f"    → {c}")
    print("\n  Rapport : rapport_lot_f.json\n")


if __name__ == "__main__":
    sys.exit(main())
