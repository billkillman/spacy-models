"""Tests du générateur de lots : non-régression des lots livrés, et construction du lot F."""
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

import generer_lots as g

#: Empreintes des lots A à D livrés en NUM-110 : ils ne doivent jamais changer.
EMPREINTES_LIVREES = {"A": "0449db5b77da", "B": "c79a882424d8", "C": "4fb54f925125", "D": "bae8d70a28ef"}


def test_lots_a_a_d_inchanges(tmp_path):
    subprocess.run([sys.executable, str(Path(g.__file__)), "--sans-controle", "--reference", str(tmp_path / "rien"),
                    "--sortie", str(tmp_path)], check=True, capture_output=True)
    manifeste = json.loads((tmp_path / "manifest_lots.json").read_text(encoding="utf-8"))
    assert {k: v["empreinte"] for k, v in manifeste["lots"].items()} == EMPREINTES_LIVREES


# ------------------------------------------------------------------ outils du lot F
@pytest.mark.parametrize("texte, attendue", [
    ("Je n'ai toujours pas reçu ma carte.", 0),
    ("Vous nous faites part de votre souhait d'obtenir un PERP.", 1),
    ("Notre cliente a effectué un virement.", 1),
    ("Madame accuse la banque de faire du forcing.", 1),
    ("Madame, Monsieur, je conteste ces frais.", 0),
])
def test_voix(texte, attendue):
    assert g.voix(texte) == attendue


def test_ajouter_contexte_ponctuation():
    assert g.ajouter_contexte("Frais injustifiés", ["Je vis au Japon."]) == "Frais injustifiés. Je vis au Japon."
    assert g.ajouter_contexte("Frais injustifiés !  ", ["Je vis au Japon."]) == "Frais injustifiés ! Je vis au Japon."


def test_contextes_neutres_vis_a_vis_des_familles():
    """Aucun contexte ne doit employer le vocabulaire d'un motif : il déplacerait les familles."""
    interdits = ("frais", "carte", "virement", "prélèvement", "compte bancaire", "application", "relevé",
                 "agence", "conseiller", "chèque", "cotisation", "bénéficiaire", "découvert")
    phrases = [p for paire in g.CONTEXTES.values() for p in paire] + [p for paire in g.PRECISIONS for p in paire]
    fautives = [(p, m) for p in phrases for m in interdits if m in p.lower()]
    assert fautives == []


# ------------------------------------------------------------------ lot F
def controle(textes):
    return [{"claimId": f"LOT-E-{i:03d}", "source_id": f"RECLA-{i:04d}", "famille": "Carte" if i % 2 else "Tarification",
             "claimVerbatim": t} for i, t in enumerate(textes)]


TEXTES = ["Ma carte n'est toujours pas arrivée depuis trois semaines."] * 0 + [
    f"Je conteste les frais prélevés sur mon compte le {i} du mois dernier." for i in range(1, 25)] + [
    "Vous nous faites part de votre mécontentement concernant votre agence.",
    "Hello, I ordered a new card three weeks ago and I still have not received it."]


def test_lot_f_apparie_et_familles_conservees():
    e = controle(TEXTES)
    f = g.lot_f(e, 1)
    assert len(f) == len(e)
    for source, cible in zip(e, f):
        assert cible["source_lot_e"] == source["claimId"] and cible["famille"] == source["famille"]
        assert cible["claimVerbatim"].startswith(source["claimVerbatim"].rstrip("."))


def test_lot_f_non_francophone_inchange():
    f = g.lot_f(controle(TEXTES), 1)
    anglais = f[-1]
    assert anglais["contexte"] == g.SANS_CONTEXTE and anglais["claimVerbatim"] == TEXTES[-1]


def test_lot_f_contextes_equilibres_et_voix():
    f = g.lot_f(controle(TEXTES), 1)
    repartition = Counter(x["contexte"] for x in f if x["contexte"] != g.SANS_CONTEXTE)
    assert max(repartition.values()) - min(repartition.values()) <= 1
    tiers = next(x for x in f if x["claimVerbatim"].startswith("Vous nous faites part"))
    assert tiers["voix"] == "tiers" and "titulaire du compte" in tiers["claimVerbatim"]


def test_lot_f_intensite_et_reproductibilite():
    e = controle(TEXTES)
    un, deux = g.lot_f(e, 1), g.lot_f(e, 2)
    assert g.lot_f(e, 1) == un                                        # graine fixe
    ajout = lambda x, s: len(x["claimVerbatim"]) - len(s["claimVerbatim"])
    assert all(ajout(b, s) > ajout(a, s) for a, b, s in zip(un, deux, e) if a["contexte"] != g.SANS_CONTEXTE)


def test_verifier_detecte_un_lot_f_non_apparie():
    e = g.numeroter(controle(TEXTES), "E")
    f = g.numeroter(g.lot_f(e, 1), "F")
    f[0]["claimVerbatim"] = "Texte sans rapport avec sa source."
    problemes = g.verifier({"A": [], "B": [], "C": [], "D": [], "E": e, "F": f}, set())
    assert any("ne prolonge pas" in p for p in problemes)
