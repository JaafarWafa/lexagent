"""Generate the fictional test contracts used by the evals.

    python data/make_test_contracts.py

bail_abusif.pdf: an unfurnished lease with deliberate legal problems (see
evals/gold/bail_abusif.yaml for the expected flags), mixed with ordinary lawful clauses
so the compliance agent is also tested on precision. Longer than bail_exemple.pdf, so
retrieval has to pick among more chunks. Everything in it is invented.
"""
from pathlib import Path

from fpdf import FPDF

OUT = Path(__file__).resolve().parent / "contracts"

BAIL_ABUSIF = [
    ("CONTRAT DE LOCATION - LOGEMENT VIDE", None),
    (None, "Exemple fictif créé pour tester LexAgent. Ce bail contient volontairement des clauses "
           "illegales. Toutes les personnes et données sont inventées."),
    ("I. Désignation des parties", None),
    (None, "Le présent contrat est conclu entre : M. Henri Duval, demeurant 3 place Bellecour, 69002 Lyon, "
           "agissant en son nom personnel, ci-après dénommé le bailleur ; et Mme Inès Moreau, ci-après "
           "dénommée la locataire."),
    ("II. Objet du contrat", None),
    (None, "Le présent contrat a pour objet la location d'un logement situé au 17 rue Paul Bert, 69003 Lyon, "
           "appartement n° 4, 2e étage, à usage exclusif d'habitation principale. Le logement comprend deux "
           "pièces principales, une cuisine, une salle de bains et un WC separe. Il est loué non meublé. "
           "Le locataire dispose en outre d'une cave numérotée 4 au sous-sol."),
    (None, "Équipements à usage privatif : chaudière individuelle au gaz, plaques de cuisson, hotte. "
           "Parties et équipements communs : local à vélos, local poubelles, interphone."),
    ("III. Date de prise d'effet et durée du contrat", None),
    (None, "Le contrat prend effet le 1er mars 2026. Il est conclu pour une durée d'un an, soit 12 mois. "
           "À défaut de congé, il est reconduit tacitement pour une durée d'un an."),
    ("IV. Conditions financieres", None),
    (None, "Le loyer mensuel est fixé à 800 euros hors charges. Les charges récupérables font l'objet "
           "d'une provision mensuelle de 70 euros, avec régularisation annuelle sur justificatifs. "
           "Le loyer et les charges sont payables d'avance le 1er de chaque mois."),
    (None, "Le paiement du loyer s'effectue obligatoirement par prélèvement automatique sur le compte "
           "bancaire du locataire ; aucun autre mode de paiement ne sera accepté."),
    (None, "Révision du loyer : le loyer sera révisé chaque année au 1er mars en fonction de la variation "
           "de l'indice du coût de la construction (ICC) publié par l'INSEE."),
    (None, "Tout retard de paiement entraînera une pénalité de 30 euros par jour de retard, due de plein "
           "droit sans mise en demeure préalable."),
    (None, "L'envoi de chaque quittance de loyer sera facturé 5 euros au locataire."),
    ("V. Dépôt de garantie", None),
    (None, "Le dépôt de garantie est fixé à 1600 euros, soit deux mois de loyer hors charges. Il sera "
           "restitué dans un délai d'un mois à compter de la remise des clés lorsque l'état des lieux de "
           "sortie est conforme à l'état des lieux d'entree, et de deux mois dans le cas contraire, "
           "déduction faite des sommes dues."),
    ("VI. État des lieux et diagnostics", None),
    (None, "Un état des lieux est établi contradictoirement par les parties lors de la remise et de la "
           "restitution des clés. Le dossier de diagnostic technique est annexé au présent contrat : "
           "diagnostic de performance énergétique, constat de risque d'exposition au plomb, état de "
           "l'installation intérieure de gaz et d'électricité, état des risques et pollutions."),
    ("VII. Congé", None),
    (None, "La locataire peut donner congé à tout moment en respectant un préavis de six mois, par lettre "
           "recommandée avec accusé de réception, acte de commissaire de justice ou remise en main propre "
           "contre recepisse."),
    (None, "Le bailleur peut donner congé à l'échéance du bail en respectant un préavis de trois mois."),
    ("VIII. Obligations du locataire", None),
    (None, "La locataire est tenue de payer le loyer et les charges aux termes convenus, d'user "
           "paisiblement des locaux, de prendre en charge l'entretien courant du logement et les menues "
           "réparations, et de ne pas transformer les locaux sans l'accord écrit du bailleur."),
    (None, "La locataire doit s'assurer contre les risques locatifs auprès de la compagnie Assur'Habitat "
           "Lyon, désignée par le bailleur, et en justifier chaque année."),
    (None, "La détention de tout animal, quel qu'il soit, est formellement interdite dans le logement."),
    (None, "La locataire s'interdit d'héberger, même temporairement, toute personne ne figurant pas au "
           "présent contrat."),
    (None, "La sous-location, totale ou partielle, est interdite sauf accord écrit du bailleur, y compris "
           "sur le prix du loyer."),
    (None, "En vue de la vente ou de la relocation du logement, la locataire devra laisser visiter le "
           "logement tous les jours, y compris les dimanches et jours fériés, de 8 heures à 20 heures."),
    ("IX. Obligations du bailleur", None),
    (None, "Le bailleur est tenu de remettre à la locataire un logement décent, en bon état d'usage et de "
           "réparation, d'assurer la jouissance paisible du logement, et d'entretenir les locaux en état "
           "de servir à l'usage prévu par le contrat. Il effectue les réparations autres que locatives."),
    (None, "Le bailleur remet gratuitement une quittance au locataire qui en fait la demande."),
    ("X. Travaux", None),
    (None, "La locataire doit permettre l'accès aux lieux loués pour la préparation et l'exécution de "
           "travaux d'amélioration des parties communes ou privatives, ou de travaux necessaires au "
           "maintien en état du logement. Les travaux ne peuvent avoir lieu les samedis, dimanches et "
           "jours fériés sans l'accord de la locataire."),
    ("XI. Clause résolutoire", None),
    (None, "Le bail sera résilié de plein droit en cas de défaut de paiement du loyer, des charges ou du "
           "dépôt de garantie, six semaines après un commandement de payer demeuré infructueux, ou en cas "
           "de défaut d'assurance."),
    (None, "Le bail sera également résilié de plein droit en cas de non-respect du règlement intérieur de "
           "l'immeuble par la locataire."),
    ("XII. Annexes", None),
    (None, "Sont annexes au présent contrat : le dossier de diagnostic technique, l'état des lieux "
           "d'entree, la notice d'information, et le règlement de copropriété (extraits)."),
    (None, "Fait à Lyon, le 10 février 2026, en deux exemplaires."),
]


FONTS = [  # a Unicode TTF font, for French accents (core PDF fonts are latin-1 only)
    ("C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/arialbd.ttf"),
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ("/Library/Fonts/Arial.ttf", "/Library/Fonts/Arial Bold.ttf"),
]


def _font(pdf: FPDF) -> str:
    for regular, bold in FONTS:
        if Path(regular).exists() and Path(bold).exists():
            pdf.add_font("Body", "", regular)
            pdf.add_font("Body", "B", bold)
            return "Body"
    raise SystemExit("No Unicode TTF font found; add one to FONTS.")


def write_pdf(blocks, path: Path, paragraphs_per_page: int = 5) -> None:
    pdf = FPDF(format="A4")
    pdf.set_margins(22, 22, 22)
    pdf.set_auto_page_break(True, margin=22)
    font = _font(pdf)
    pdf.add_page()
    count = 0
    for heading, text in blocks:
        if heading:
            if count >= paragraphs_per_page:  # short pages, so the contract spans several pages
                pdf.add_page()
                count = 0
            pdf.set_font(font, "B", 12)
            pdf.multi_cell(0, 7, heading, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(1)
        else:
            pdf.set_font(font, "", 11)
            pdf.multi_cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
            pdf.ln(3)
            count += 1
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(path))


if __name__ == "__main__":
    write_pdf(BAIL_ABUSIF, OUT / "bail_abusif.pdf")
    print(f"wrote {OUT / 'bail_abusif.pdf'}")
