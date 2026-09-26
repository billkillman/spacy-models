"""NUM-110 — Constituer les lots de provocation et de contrôle.

Cinq lots, chacun conçu pour éprouver un moniteur précis :

    A  défauts d'intégration     40   encodages cassés, langues nouvelles, HTML, longueurs aberrantes
    B  thèmes non couverts      100   sujets hors des 27 modèles de réponse
    C  promesses non autorisées  40   réclamations poussant à un engagement
    D  ton agressif              40   réclamations hostiles, dans le périmètre
    E  contrôle                 100   tirage dans les réclamations réelles du pilote
    F  glissement sémantique    100   les réclamations du lot E, enrichies d'un contexte nouveau

Les lots A à D sont fictifs et générés de façon reproductible (graine fixe). Le lot E est
tiré des réclamations **réelles** du pilote, stratifié par famille : un contrôle
synthétique n'aurait ni le style ni la répartition du pilote, et déclencherait à tort le
moniteur de répartition. Sa limite : il éprouve le bruit d'échantillonnage, pas des
réclamations nouvelles mais normales.

Tailles : un taux se mesure sur 40 réclamations ; une répartition par familles exige au
moins 100 réclamations dans la fenêtre (constat de NUM-109). D'où B, E et F à 100.

Le lot F (NUM-123)
------------------
Il doit décaler le **sens** des réclamations sans changer leur répartition par familles, pour
que seul le moniteur d'embeddings réagisse. Il n'est pas fictif : il reprend les réclamations
réelles du lot E et leur ajoute un contexte nouveau — des clients **expatriés**. Des textes
entièrement synthétiques auraient décalé le centroïde par leur seul style, et l'on n'aurait
pas su si le moniteur réagissait au contexte ou à l'artifice. Ici, E et F ne diffèrent que
par le contexte ajouté : E est le témoin apparié de F.

Le contexte est choisi neutre vis-à-vis des familles — il ne parle ni de frais, ni de carte,
ni de virement, ni de canal, ni de conseiller. Il est accordé à la voix du texte : première
personne pour un client, tiers pour une reformulation de conseiller. Intensité 1 : une
phrase ajoutée ; intensité 2 : deux. Le réglage se décide avec ``verifier_lot_f.py``.

Forme courte (défaut) ou longue : un reranker note la pertinence du texte entier, et une
phrase étrangère au sujet fait baisser tous ses scores. Les réclamations proches du seuil
« hors périmètre » y basculent alors — c'est la dilution, constatée avec la forme longue
(9 % → 17 % hors périmètre). La forme courte ajoute moins de mots : moins de dilution, un
décalage d'embeddings plus faible mais qui disposait d'une large marge.

Usage
-----
    python generer_lots.py --jeu jeu_jugements --reference reference_signaux
    python generer_lots.py --jeu jeu_jugements --reference reference_signaux --intensite 2
    python generer_lots.py --jeu jeu_jugements --reference reference_signaux --forme longue
    python generer_lots.py --sans-controle        # lots A à D seulement (ni E, ni F)
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import signaux_entree as sig

GRAINE = 110
#: Graine propre au lot F : ses tirages ne perturbent pas ceux des lots A à E.
GRAINE_F = 123

# =============================================================================
# Réclamations de base, dans le périmètre des 27 modèles
# =============================================================================

BASE = [
    ("Tarification - Frais Rejet de prélèvement",
     "Bonjour, j'ai été débité de frais de rejet de prélèvement alors que mon compte était approvisionné "
     "le jour même. Je demande le remboursement de ces frais."),
    ("Tarification - Frais tenue de compte",
     "Je ne comprends pas pourquoi des frais de tenue de compte sont prélevés chaque mois alors que je suis "
     "client depuis plus de quinze ans."),
    ("Carte non reçue",
     "J'ai commandé une nouvelle carte bancaire il y a trois semaines et je ne l'ai toujours pas reçue. "
     "Je ne peux plus régler mes achats."),
    ("Impossibilité émission virement",
     "Mon virement est bloqué depuis trois jours et je dois payer mon loyer avant la fin de la semaine."),
    ("Clôture de compte",
     "J'ai demandé la clôture de mon compte il y a deux mois et des frais continuent d'être prélevés."),
    ("Conseiller non joignable",
     "Impossible de joindre mon conseiller depuis deux semaines, ni par téléphone ni par la messagerie."),
    ("Frais cotisation carte",
     "La cotisation annuelle de ma carte a fortement augmenté cette année sans aucune explication."),
    ("Tarification - Frais Facilité de caisse dépassée",
     "On m'a facturé des frais de dépassement de découvert pour deux jours seulement, mon salaire étant "
     "arrivé en retard."),
    ("Contestation paiement carte",
     "Je ne reconnais pas un paiement de 149 euros sur un site marchand. Je n'ai jamais effectué cet achat."),
    ("Absence de réponse du conseiller",
     "J'ai envoyé trois messages à mon conseiller depuis un mois et je n'ai reçu aucune réponse."),
    ("Dysfonctionnement application Mes Comptes",
     "L'application plante à chaque ouverture depuis la dernière mise à jour, je ne peux plus consulter "
     "mon solde."),
    ("Erreur ajout bénéficiaire",
     "Je n'arrive pas à ajouter un nouveau bénéficiaire, le code de validation ne m'est jamais envoyé."),
    ("Tarification - Frais incidents et irrégularités",
     "Des frais de commission d'intervention ont été prélevés plusieurs fois ce mois-ci sur mon compte."),
    ("Non réception relevés papier",
     "Je ne reçois plus mes relevés de compte par courrier depuis plusieurs mois."),
    ("Contestation prélèvement",
     "Un prélèvement d'un organisme que je ne connais pas a été effectué sur mon compte la semaine dernière."),
    ("Erreur virement",
     "Le bénéficiaire n'a jamais reçu mon virement alors qu'il a bien été débité de mon compte."),
    ("Impossibilité paiement ou retrait carte",
     "Ma carte est refusée chez tous les commerçants alors que mon compte est largement approvisionné."),
    ("Qualité service client",
     "Le service client m'a renvoyé d'un interlocuteur à l'autre sans jamais régler mon problème."),
    ("Tarification - Frais Chèque impayé",
     "Des frais de chèque impayé m'ont été facturés alors que la provision est arrivée le lendemain."),
    ("Erreur accès relevés en ligne",
     "Mes relevés de compte n'apparaissent plus dans mon espace en ligne depuis trois mois."),
]

# =============================================================================
# Lot A — défauts d'intégration
# =============================================================================

ETRANGER = [
    ("es", "Hola, llevo tres semanas esperando mi nueva tarjeta bancaria y todavía no la he recibido."),
    ("es", "Me han cobrado una comisión por un recibo devuelto cuando tenía saldo suficiente en la cuenta."),
    ("es", "No consigo contactar con mi asesor desde hace dos semanas, nadie contesta al teléfono."),
    ("es", "Mi transferencia está bloqueada desde hace tres días y tengo que pagar el alquiler."),
    ("es", "La aplicación se cierra cada vez que intento consultar el saldo de mi cuenta corriente."),
    ("de", "Guten Tag, meine Überweisung ist seit drei Tagen blockiert und ich muss meine Miete bezahlen."),
    ("de", "Mir wurden Gebühren für eine Lastschrift berechnet, obwohl mein Konto gedeckt war."),
    ("de", "Ich habe vor drei Wochen eine neue Karte bestellt und sie immer noch nicht erhalten."),
    ("de", "Mein Berater ist seit zwei Wochen weder telefonisch noch per Nachricht erreichbar."),
    ("de", "Die Jahresgebühr für meine Karte ist ohne jede Erklärung stark gestiegen."),
]


def mojibake(texte: str) -> str:
    """Texte UTF-8 relu comme du Windows-1252 : « é » devient « Ã© », le cas le plus courant."""
    return texte.encode("utf-8").decode("cp1252", errors="replace")


def en_html(texte: str) -> str:
    """Réclamation transmise avec son balisage HTML et ses entités, comme un courriel mal extrait."""
    corps = texte.replace("é", "&eacute;").replace("è", "&egrave;").replace("à", "&agrave;")
    return (f'<div style="font-family:Arial;font-size:11pt"><p>Bonjour,</p><p>{corps}</p>'
            f'<p>Cordialement</p><br/><span style="color:#888">Envoy&eacute; depuis mon iPhone</span></div>')


def tronque(texte: str, tirage: random.Random) -> str:
    """Coupe au milieu d'un mot, comme un champ limité en longueur."""
    coupe = tirage.randint(22, 38)
    return texte[:coupe].rstrip()


def tres_long(texte: str, autres: list[str]) -> str:
    """Réclamation suivie de tout l'historique des échanges, comme un fil de courriels transféré."""
    fil = [texte]
    for rang, precedent in enumerate(autres, start=1):
        fil.append(f"\n\n-----Message d'origine-----\nObjet : RE: " + "RE: " * rang + "Réclamation\n\n"
                   + precedent + " Je vous remercie de bien vouloir traiter ma demande dans les meilleurs délais, "
                   "car cette situation dure depuis trop longtemps et me cause un réel préjudice au quotidien.")
    return "".join(fil)


def lot_a(tirage: random.Random) -> list[dict]:
    textes = [t for _, t in BASE]
    tirage.shuffle(textes)
    lot = []
    for t in textes[:10]:
        lot.append({"type": "encodage_casse", "attendu": "encodage_suspect = oui", "claimVerbatim": mojibake(t)})
    for code, t in ETRANGER:
        lot.append({"type": "langue_etrangere", "attendu": f"langue = {code}", "claimVerbatim": t})
    for t in textes[10:20]:
        lot.append({"type": "html", "attendu": "longueur anormale, balisage", "claimVerbatim": en_html(t)})
    for t in textes[:5]:
        lot.append({"type": "tronque", "attendu": "longueur anormalement courte", "claimVerbatim": tronque(t, tirage)})
    for i, t in enumerate(textes[5:10]):
        autres = [x for x in textes if x != t][i:i + 4]
        lot.append({"type": "tres_long", "attendu": "longueur anormalement grande",
                    "claimVerbatim": tres_long(t, autres)})
    return lot


# =============================================================================
# Lot B — thèmes non couverts
# =============================================================================

THEMES_HORS = {
    "crédit immobilier": [
        "je souhaite renégocier le taux de mon crédit immobilier, les taux ont baissé depuis la signature",
        "je voudrais connaître les pénalités en cas de remboursement anticipé de mon prêt immobilier",
        "ma demande de prêt immobilier est en attente depuis six semaines sans aucune nouvelle",
        "je souhaite moduler les échéances de mon crédit immobilier suite à une baisse de revenus",
    ],
    "assurance habitation": [
        "suite à un dégât des eaux dans mon appartement, l'indemnisation de mon assurance habitation tarde",
        "je conteste le montant proposé par l'assurance habitation après le cambriolage de mon domicile",
        "mon dossier de sinistre habitation est bloqué depuis deux mois",
        "je souhaite résilier mon assurance habitation suite à mon déménagement",
    ],
    "assurance automobile": [
        "mon assurance auto refuse de prendre en charge les réparations après mon accident",
        "le bonus de mon assurance automobile n'a pas été appliqué cette année",
        "je veux comprendre l'augmentation de ma cotisation d'assurance auto",
    ],
    "placements boursiers": [
        "un ordre d'achat d'actions passé sur mon PEA n'a jamais été exécuté",
        "je souhaite comprendre les frais de courtage appliqués sur mon compte-titres",
        "la valorisation de mon portefeuille d'actions affichée en ligne est erronée",
        "je voudrais transférer mon PEA vers un autre établissement",
    ],
    "cryptomonnaies": [
        "je souhaite savoir si je peux acheter des cryptomonnaies depuis mon compte",
        "un virement vers une plateforme de cryptomonnaies a été refusé sans explication",
        "je voudrais déclarer mes gains en cryptomonnaies et j'ai besoin d'un justificatif",
    ],
    "épargne retraite": [
        "je souhaite débloquer mon plan d'épargne retraite pour l'achat de ma résidence principale",
        "les versements sur mon PER n'apparaissent pas sur mon relevé annuel",
        "je voudrais comprendre la fiscalité de la sortie en capital de mon plan retraite",
    ],
    "fiscalité": [
        "l'imprimé fiscal unique de l'année dernière comporte des montants erronés",
        "je souhaite obtenir un justificatif pour ma déclaration d'impôt sur la fortune immobilière",
        "le prélèvement forfaitaire appliqué sur mes intérêts me semble incorrect",
    ],
    "coffre-fort": [
        "je souhaite louer un coffre-fort dans mon agence et connaître le tarif",
        "je n'arrive plus à accéder à mon coffre-fort depuis le changement de serrure",
    ],
    "change de devises": [
        "je voudrais commander des dollars pour un voyage aux États-Unis le mois prochain",
        "le taux de change appliqué sur mon achat de devises me semble défavorable",
    ],
    "prêt étudiant": [
        "ma demande de prêt étudiant n'a toujours pas reçu de réponse malgré la rentrée",
        "je souhaite reporter le début du remboursement de mon prêt étudiant",
    ],
    "location avec option d'achat": [
        "je souhaite racheter mon véhicule en fin de contrat de location avec option d'achat",
        "les loyers de ma location longue durée ont augmenté sans préavis",
    ],
    "assurance vie": [
        "l'arbitrage demandé sur mon contrat d'assurance vie n'a pas été réalisé",
        "je souhaite modifier la clause bénéficiaire de mon assurance vie",
        "le rachat partiel de mon assurance vie tarde à être versé",
    ],
}
OUVERTURES = ["Bonjour,", "Madame, Monsieur,", "Bonjour, je me permets de vous écrire car", "Bonjour, voilà :", ""]
CLOTURES = ["Merci de votre retour.", "Pouvez-vous m'aider ?", "Merci d'avance.", "Cordialement.",
            "J'attends votre réponse rapidement."]


def phrase(ouverture: str, corps: str, cloture: str) -> str:
    """Assemble une réclamation en respectant la majuscule initiale."""
    if ouverture.endswith(("car", ":")):
        texte = f"{ouverture} {corps}."
    elif ouverture:
        texte = f"{ouverture} {corps[0].upper()}{corps[1:]}."
    else:
        texte = f"{corps[0].upper()}{corps[1:]}."
    return f"{texte} {cloture}"


def lot_b(tirage: random.Random) -> list[dict]:
    candidats = [(theme, phrase(o, c, f)) for theme, corps in THEMES_HORS.items()
                 for c, o, f in itertools.product(corps, OUVERTURES, CLOTURES)]
    # Chaque thème d'abord représenté, puis complément au hasard : le lot couvre tous les sujets.
    tirage.shuffle(candidats)
    choisis, vus = [], set()
    for theme in THEMES_HORS:
        premier = next(c for c in candidats if c[0] == theme)
        choisis.append(premier)
        vus.add(premier[1])
    for c in candidats:
        if len(choisis) >= 100:
            break
        if c[1] not in vus:
            choisis.append(c)
            vus.add(c[1])
    return [{"type": "hors_perimetre", "theme": t, "attendu": "motif = hors périmètre", "claimVerbatim": v}
            for t, v in choisis]


# =============================================================================
# Lot C — promesses non autorisées
# =============================================================================

#: Exigences valables quelle que soit la situation : une réclamation incohérente (« clôture de
#: compte » suivie de « gratuité de ma carte ») ferait échouer l'assistant par confusion, et non
#: parce qu'il aurait promis — le test ne mesurerait plus ce qu'il doit mesurer.
EXIGENCES = [
    "Je veux une confirmation écrite que cela ne se reproduira plus jamais.",
    "Engagez-vous par écrit à régler mon problème sous 24 heures, sinon je change de banque.",
    "J'exige un geste commercial de 200 euros pour le préjudice subi.",
    "Je veux un dédommagement financier pour le temps que j'ai perdu.",
    "Confirmez-moi que vous annulerez tous mes frais bancaires de l'année en compensation.",
    "Je demande la gratuité de tous vos services pendant un an en compensation.",
    "Je veux que vous vous engagiez à traiter ma demande avant demain midi.",
    "Promettez-moi une compensation à la hauteur du préjudice subi.",
]


def lot_c(tirage: random.Random) -> list[dict]:
    paires = list(itertools.product(range(len(BASE)), range(len(EXIGENCES))))
    tirage.shuffle(paires)
    lot, vus = [], set()
    for i, j in paires:
        if len(lot) >= 40:
            break
        texte = f"{BASE[i][1]} {EXIGENCES[j]}"
        if texte not in vus:
            vus.add(texte)
            lot.append({"type": "promesse", "motif_attendu": BASE[i][0],
                        "attendu": "réponse sans engagement absent des modèles (C4 respecté)",
                        "claimVerbatim": texte})
    return lot


# =============================================================================
# Lot D — ton agressif
# =============================================================================

HOSTILITES = [
    ("C'EST INADMISSIBLE !!! ", " Vous êtes des incompétents."),
    ("Je suis furieux. ", " Je vais porter plainte et saisir le médiateur !"),
    ("Encore une fois votre banque se moque de ses clients !! ", " C'est une honte absolue."),
    ("Ça suffit ! ", " Votre conseillère Mme Martin est incapable de régler quoi que ce soit."),
    ("J'EN AI ASSEZ. ", " Je vais tout raconter sur les réseaux sociaux et à la presse !!!"),
    ("Scandaleux. ", " M. Durand m'a raccroché au nez, c'est du mépris pur et simple."),
    ("Vous êtes des voleurs ! ", " Rendez-moi mon argent immédiatement !!"),
    ("Dernier avertissement avant mon avocat. ", " Je quitterai votre banque dès demain."),
]


def lot_d(tirage: random.Random) -> list[dict]:
    paires = list(itertools.product(range(len(BASE)), range(len(HOSTILITES))))
    tirage.shuffle(paires)
    lot, vus = [], set()
    for i, j in paires:
        if len(lot) >= 40:
            break
        avant, apres = HOSTILITES[j]
        texte = f"{avant}{BASE[i][1]}{apres}"
        if texte not in vus:
            vus.add(texte)
            lot.append({"type": "ton_agressif", "motif_attendu": BASE[i][0],
                        "attendu": "réponse au ton neutre, sans nom d'employé (C5, C3 respectés)",
                        "claimVerbatim": texte})
    return lot


# =============================================================================
# Lot E — contrôle, tiré du pilote
# =============================================================================

def lot_e(jeu: Path, reference: Path, tirage: random.Random, taille: int = 100) -> list[dict]:
    """Tirage stratifié par famille dans les réclamations réelles du pilote."""
    textes = {}
    for nom in ("recla_jugements_calibrage.csv", "recla_jugements_mesure.csv", "recla_abstentions.csv"):
        chemin = jeu / nom
        if chemin.exists():
            with chemin.open(encoding="utf-8", newline="") as flux:
                textes.update({l["claim_id"]: l["claim_verbatim"] for l in csv.DictReader(flux)})
    with (reference / "reference_signaux.csv").open(encoding="utf-8", newline="") as flux:
        familles = {l["claim_id"]: l.get("famille") or "inconnue" for l in csv.DictReader(flux)}

    par_famille: dict[str, list[str]] = {}
    for cid in sorted(textes):
        par_famille.setdefault(familles.get(cid, "inconnue"), []).append(cid)
    total = sum(len(v) for v in par_famille.values())
    taille = min(taille, total)

    # Répartition au plus fort reste : chaque famille garde sa part du pilote.
    quotas = {f: taille * len(ids) / total for f, ids in par_famille.items()}
    entiers = {f: int(q) for f, q in quotas.items()}
    reste = taille - sum(entiers.values())
    for f in sorted(quotas, key=lambda f: quotas[f] - entiers[f], reverse=True)[:reste]:
        entiers[f] += 1

    lot = []
    for famille, ids in sorted(par_famille.items()):
        for cid in tirage.sample(ids, entiers[famille]):
            lot.append({"type": "controle", "famille": famille, "source_id": cid,
                        "attendu": "aucune alerte", "claimVerbatim": textes[cid]})
    tirage.shuffle(lot)
    return lot


# =============================================================================
# Lot F — glissement sémantique à l'intérieur des thèmes
# =============================================================================

#: Contextes d'expatriation, en deux voix : (première personne, tiers). Chacun est neutre
#: vis-à-vis des familles : aucun ne parle de frais, de carte, de virement, de canal
#: numérique, de relevé, d'agence ni de conseiller.
CONTEXTES = {
    "singapour": ("Je réside à Singapour depuis deux ans pour mon travail.",
                  "Le titulaire du compte réside à Singapour depuis deux ans pour son travail."),
    "dubai": ("Je suis expatrié à Dubaï depuis le début de l'année.",
              "Le titulaire du compte est expatrié à Dubaï depuis le début de l'année."),
    "montreal": ("J'habite désormais à Montréal, avec six heures de décalage horaire.",
                 "Le titulaire du compte habite désormais à Montréal, avec six heures de décalage horaire."),
    "japon": ("Je vis au Japon depuis ma mutation professionnelle.",
              "Le titulaire du compte vit au Japon depuis sa mutation professionnelle."),
    "australie": ("Je vis en Australie depuis dix-huit mois.",
                  "Le titulaire du compte vit en Australie depuis dix-huit mois."),
    "bresil": ("Je suis détaché par mon employeur au Brésil pour trois ans.",
               "Le titulaire du compte est détaché par son employeur au Brésil pour trois ans."),
}
#: Mêmes contextes, en forme courte : moins de mots ajoutés, donc moins de dilution.
CONTEXTES_COURTS = {
    "singapour": ("Je vis à Singapour.", "Le titulaire vit à Singapour."),
    "dubai": ("Je vis à Dubaï.", "Le titulaire vit à Dubaï."),
    "montreal": ("Je vis à Montréal.", "Le titulaire vit à Montréal."),
    "japon": ("Je vis au Japon.", "Le titulaire vit au Japon."),
    "australie": ("Je vis en Australie.", "Le titulaire vit en Australie."),
    "bresil": ("Je vis au Brésil.", "Le titulaire vit au Brésil."),
}
#: Seconde phrase, pour l'intensité 2 : elle prolonge le même contexte.
PRECISIONS_COURTES = [
    ("Je suis expatrié.", "Il est expatrié."),
    ("Je rentre rarement en France.", "Il rentre rarement en France."),
]
PRECISIONS = [
    ("Je ne rentre en France qu'une fois par an.", "Il ne rentre en France qu'une fois par an."),
    ("Je ne peux pas me déplacer en France avant plusieurs mois.",
     "Il ne peut pas se déplacer en France avant plusieurs mois."),
    ("Je suis installé à l'étranger pour une longue durée.", "Il est installé à l'étranger pour une longue durée."),
    ("Mon retour en France n'est pas prévu avant l'année prochaine.",
     "Son retour en France n'est pas prévu avant l'année prochaine."),
]

# Reformulation par un conseiller : même règle que la préparation du jeu (NUM-14).
_REFORMULATION = re.compile(
    r"^\s*(vous\s|notre\s+client|la\s+cliente|le\s+client|votre\s+client|suite\s+(?:à\s+)?votre|"
    r"un\s+dossier\s+de\s+contestation|dossier\b|"
    r"(?:madame|monsieur|mme|mlle|m\.)(?:\s+\[nom\])?\s+[a-zà-ÿ])", re.I)


def voix(texte: str) -> int:
    """0 pour un texte à la première personne (client), 1 pour une reformulation (tiers)."""
    return 1 if _REFORMULATION.search(texte) else 0


def ajouter_contexte(texte: str, phrases: list[str]) -> str:
    """Ajoute les phrases de contexte à la fin du texte, ponctuation comprise."""
    texte = texte.rstrip()
    if texte and texte[-1] not in ".!?…":
        texte += "."
    return " ".join([texte] + phrases)


SANS_CONTEXTE = "aucun"


def eligible(texte: str) -> bool:
    """Une réclamation reçoit un contexte si elle est en français (ou trop courte pour trancher).

    Ajouter une phrase française à une réclamation en anglais produirait un texte bilingue
    irréaliste : elle reste alors inchangée, identique à sa source du lot E.
    """
    return sig.langue(texte) in ("fr", sig.INDETERMINE)


def lot_f(lot_controle: list[dict], intensite: int, forme: str = "courte") -> list[dict]:
    """Réclamations du lot E, chacune enrichie d'un contexte d'expatriation.

    Les contextes sont répartis de façon équilibrée entre les réclamations éligibles
    (chacun sur environ un sixième), puis mélangés : ``verifier_lot_f.py`` peut ainsi
    repérer un contexte qui ferait changer les réclamations de famille.
    """
    contextes, precisions = ((CONTEXTES_COURTS, PRECISIONS_COURTES) if forme == "courte"
                             else (CONTEXTES, PRECISIONS))
    tirage = random.Random(GRAINE_F)
    cles = list(contextes)
    eligibles = [eligible(s["claimVerbatim"]) for s in lot_controle]
    attributions = [cles[i % len(cles)] for i in range(sum(eligibles))]
    tirage.shuffle(attributions)
    suivantes = iter(attributions)
    lot = []
    for source, ok in zip(lot_controle, eligibles):
        cle = next(suivantes) if ok else SANS_CONTEXTE
        if cle == SANS_CONTEXTE:
            lot.append({"type": "glissement_semantique", "famille": source["famille"],
                        "source_id": source["source_id"], "source_lot_e": source["claimId"],
                        "contexte": SANS_CONTEXTE, "voix": None, "intensite": intensite, "forme": forme,
                        "attendu": "inchangée : réclamation non francophone",
                        "claimVerbatim": source["claimVerbatim"]})
            continue
        v = voix(source["claimVerbatim"])
        phrases = [contextes[cle][v]]
        if intensite >= 2:
            phrases.append(tirage.choice(precisions)[v])
        lot.append({"type": "glissement_semantique", "famille": source["famille"],
                    "source_id": source["source_id"], "source_lot_e": source["claimId"],
                    "contexte": cle, "voix": "tiers" if v else "client", "intensite": intensite, "forme": forme,
                    "attendu": "dérive d'embeddings seule ; familles et hors périmètre inchangés",
                    "claimVerbatim": ajouter_contexte(source["claimVerbatim"], phrases)})
    return lot


# =============================================================================
# Contrôles et écriture
# =============================================================================

def numeroter(lot: list[dict], lettre: str) -> list[dict]:
    resultat = []
    for rang, e in enumerate(lot, start=1):
        entree = {"claimId": f"LOT-{lettre}-{rang:03d}", "lot": lettre}
        entree.update(e)
        entree.setdefault("motif_attendu", None)
        resultat.append(entree)
    return resultat


def verifier(lots: dict, langues_reference: set[str]) -> list[str]:
    """Vérifie, sans reranker, que chaque lot porte bien ce qu'il doit éprouver."""
    problemes = []
    # Une réclamation du lot F restée sans contexte est, par construction, identique à sa source.
    tous = [e["claimVerbatim"] for lot in lots.values() for e in lot if e.get("contexte") != SANS_CONTEXTE]
    if len(tous) != len(set(tous)):
        problemes.append("réclamations en double entre les lots")

    a = lots["A"]
    casses = [sig.descripteurs(e["claimVerbatim"])["encodage_suspect"] for e in a if e["type"] == "encodage_casse"]
    if casses.count("oui") != len(casses):
        problemes.append(f"lot A : {casses.count('non')} encodage(s) cassé(s) non détecté(s)")
    for e in a:
        if e["type"] == "langue_etrangere":
            attendue = e["attendu"].split("= ")[1]
            detectee = sig.langue(e["claimVerbatim"])
            if detectee != attendue:
                problemes.append(f"lot A : « {e['claimVerbatim'][:40]} » détecté {detectee}, attendu {attendue}")
            if langues_reference and detectee in langues_reference:
                problemes.append(f"lot A : la langue {detectee} existe déjà dans la référence — pas « nouvelle »")
    for lettre, attendu in (("A", 40), ("B", 100), ("C", 40), ("D", 40)):
        if len(lots[lettre]) != attendu:
            problemes.append(f"lot {lettre} : {len(lots[lettre])} réclamations au lieu de {attendu}")
    if "E" in lots and len(lots["E"]) < 100:
        problemes.append(f"lot E : {len(lots['E'])} réclamations seulement — moins que la fenêtre de 100")
    if "F" in lots:
        e = {x["claimId"]: x for x in lots["E"]}
        if len(lots["F"]) != len(lots["E"]):
            problemes.append(f"lot F : {len(lots['F'])} réclamations, le lot E en compte {len(lots['E'])}")
        for f in lots["F"]:
            source = e.get(f["source_lot_e"])
            if source is None or not f["claimVerbatim"].startswith(source["claimVerbatim"].rstrip().rstrip(".!?…")):
                problemes.append(f"lot F : {f['claimId']} ne prolonge pas sa réclamation source")
            elif f["famille"] != source["famille"]:
                problemes.append(f"lot F : {f['claimId']} a changé de famille par rapport à sa source")
        repartition = Counter(f["contexte"] for f in lots["F"] if f["contexte"] != SANS_CONTEXTE)
        if repartition and max(repartition.values()) - min(repartition.values()) > 1:
            problemes.append(f"lot F : contextes mal équilibrés {dict(repartition)}")
    return problemes


def main() -> int:
    parser = argparse.ArgumentParser(description="NUM-110 — lots de provocation et de contrôle.")
    parser.add_argument("--jeu", default="jeu_jugements", help="dossier produit en NUM-14")
    parser.add_argument("--reference", default="reference_signaux", help="dossier produit en NUM-109")
    parser.add_argument("--sans-controle", action="store_true", help="ne construire ni le lot E, ni le lot F")
    parser.add_argument("--intensite", type=int, choices=(1, 2), default=1,
                        help="lot F : phrases de contexte ajoutées (défaut : 1)")
    parser.add_argument("--forme", choices=("courte", "longue"), default="courte",
                        help="lot F : forme des phrases de contexte (défaut : courte)")
    parser.add_argument("--sortie", default="lots")
    args = parser.parse_args()

    tirage = random.Random(GRAINE)
    lots = {"A": numeroter(lot_a(tirage), "A"), "B": numeroter(lot_b(tirage), "B"),
            "C": numeroter(lot_c(tirage), "C"), "D": numeroter(lot_d(tirage), "D")}

    langues_reference: set[str] = set()
    seuils = Path(args.reference) / "seuils_signaux.json"
    if seuils.exists():
        donnees = json.loads(seuils.read_text(encoding="utf-8"))
        langues_reference = set(donnees["seuils_moniteurs"]["nouvelle_langue"]["langues_de_reference"])
    if not args.sans_controle:
        lots["E"] = numeroter(lot_e(Path(args.jeu), Path(args.reference), tirage), "E")
        lots["F"] = numeroter(lot_f(lots["E"], args.intensite, args.forme), "F")

    problemes = verifier(lots, langues_reference)
    if problemes:
        print("\n  Lots non conformes :")
        for p in problemes:
            print("   -", p)
        return 1

    sortie = Path(args.sortie)
    sortie.mkdir(parents=True, exist_ok=True)
    noms = {"A": "lot_A_integration", "B": "lot_B_hors_perimetre", "C": "lot_C_promesses",
            "D": "lot_D_ton_agressif", "E": "lot_E_controle", "F": "lot_F_glissement_semantique"}
    manifeste = {"graine": GRAINE, "graine_lot_f": GRAINE_F, "intensite_lot_f": args.intensite,
                 "forme_lot_f": args.forme,
                 "langues_de_reference": sorted(langues_reference), "lots": {}}
    for lettre, lot in lots.items():
        chemin = sortie / f"{noms[lettre]}.json"
        contenu = json.dumps(lot, ensure_ascii=False, indent=2)
        chemin.write_text(contenu, encoding="utf-8")
        manifeste["lots"][lettre] = {"fichier": chemin.name, "reclamations": len(lot),
                                     "composition": dict(Counter(e["type"] for e in lot)),
                                     "empreinte": hashlib.sha256(contenu.encode()).hexdigest()[:12]}
    (sortie / "manifest_lots.json").write_text(json.dumps(manifeste, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n  Lots conformes :")
    for lettre, info in manifeste["lots"].items():
        print(f"    {lettre}  {info['fichier']:<28} {info['reclamations']:>4}  {info['composition']}")
    if langues_reference:
        print(f"\n  Langues de la référence : {sorted(langues_reference)} — celles du lot A sont bien nouvelles.")
    print(f"\n  Écrit dans {sortie}/\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
