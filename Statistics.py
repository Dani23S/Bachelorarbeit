import pandas as pd
import numpy as np
from scipy.stats import spearmanr, chi2_contingency
import matplotlib

matplotlib.use('TkAgg')

# Datenquelle
INPUT_FILE = r"Daten/AuswertungUmfrage.xlsx"

# Spaltennamen kodieren um Übersichtlichkeit zu gewährleisten
def load_and_rename_data(file_path):
    df = pd.read_excel(file_path)
    column_mapping = {
        "Ihr Alter": "age",
        "Ihr Geschlecht": "gender",
        "Ihr derzeitiger Beschäftigungsstatus": "employment_status",
        "Wie sehr fühlen Sie sich mit der Marke BMW verbunden?": "brand_connection",
        "Wie oft besuchen Sie die BMW Open?": "bmw_open_frequency",
        "Warum haben Sie die BMW Open noch nie besucht? (Mehrfachauswahl möglich)": "why_never_visited",
        "Aus welchem Grund haben Sie die BMW Open bisher besucht? (Mehrfachauswahl möglich)": "why_visited_bmw_open",
        "Haben Sie schon einmal Augmented Reality- und Virtual Reality-Technologien auf einem Event genutzt?": "ar_vr_experience",
        "Wie vertraut sind Sie mit der Augmented Reality-Technologie?": "ar_familiarity",
        "Wie vertraut sind Sie mit der Virtual Reality-Technologie?": "vr_familiarity",
        "Wie bewerten Sie die Möglichkeit, bei einem Event Augmented Reality- und Virtual Reality-Technologien zu nutzen, um Informationen über die Marke BMW zu erhalten?": "ar_vr_info_usefulness",
        "Stellen Sie sich vor, Sie könnten mithilfe von Augmented Reality digitale Inhalte direkt in Ihrer Umgebung sehen und mit ihnen interagieren. Zum Beispiel könnten Sie ein BMW-Auto auf Ihrem Smartphone oder einer Augmented Reality-Brille in 3D betrachten, d": "ar_excitement",
        "Welche Art von Augmented Reality- und Virtual Reality-Erlebnissen würde Sie bei einem BMW-Event am meisten ansprechen? (Mehrfachauswahl möglich)": "ar_vr_which_experiences",
        "Glauben Sie, dass Technologien wie Augmented Reality und Virtual Reality, z. B. für virtuelle Testfahrten oder interaktive Infos, die Marke BMW als modern und fortschrittlich darstellen können? ": "ar_vr_modern",
        "Wie bewerten Sie die Idee, bei einem Event Augmented Reality- oder Virtual Reality-Erlebnisse wie virtuelle Fahrzeugpräsentationen zu erleben?": "ar_vr_interest",
        "Stellen Sie sich vor, Sie sitzen in einem BMW-Sportwagen und nehmen mithilfe von Virtual-Reality-Technologien (VR-Brille, Kopfhörer, Auto) an einem virtuellen Rennen auf dem Nürburgring teil. Sie spüren die Vibration des Autos, können mittels Gas und Brem": "vr_emotion",
        "Wie wichtig ist es Ihnen, dass alle Elemente eines Events, wie Aktivitäten, Technologien und das Design, gut aufeinander abgestimmt sind und ein beeindruckendes, mitreißendes Erlebnis schaffen? ": "event_holistic",
        "Finden Sie, dass Augmented Reality- und Virtual Reality-Technologien eher jüngere Zielgruppen (Millennials und Gen Z) ansprechen? ": "ar_vr_younger_target",
        "Glauben Sie, dass Sie sich nach einer positiven Augmented Reality- und/oder Virtual Reality-Technologie Erfahrung mit der Marke BMW verbunden fühlen? ": "brand_connection_after_ar_vr",
        "Würden Sie nach einem positiven Augmented Reality- und/oder Virtual Reality-Erlebnis eine Testfahrt mit ihrem BMW-Wunsch-Fahrzeug vereinbaren?": "test_drive_intent",
        "Würden Sie durch positive Augmented Reality- und/oder Virtual Reality-Erfahrung die BMW Open öfter besuchen?": "visit_bmw_open_more"
    }
    df.rename(columns=column_mapping, inplace=True)
    return df

#Hilfsfunktion zur Kodierung der Daten
def encode_data(df):

    # Altersgruppen-Kodierung
    age_map = {
        "<18": 1,
        "18-24": 2,
        "25-34": 3,
        "35-44": 4,
        "45-54": 5,
        ">55": 6
    }
    df["age_numeric"] = df["age"].map(age_map)

    # Geschlechts-Kodierung
    gender_map = {
        "männlich": 1,
        "weiblich": 2,
        "divers": 3
    }
    df["gender_numeric"] = df["gender"].map(gender_map)

    # Likert-Skalen-Kodierung
    likert_map = {  # Gleiche Map wie vorher
        "Ja, auf jeden Fall": 1,
        "Ja, teilweise": 2,
        "Nein": 3,
        "Gar nicht vertraut": 4,
        "Kaum vertraut": 3,
        "Etwas vertraut": 2,
        "Sehr vertraut": 1,
        "Sehr uninteressant": 1,
        "Weniger interessant": 3,
        "Neutral": 3,
        "Eher interessant": 2,
        "Sehr interessant": 1,
        "Weniger wichtig": 3,
        "Eher wichtig": 2,
        "Sehr wichtig": 1,
        "Weniger stark": 3,
        "Eher stark": 2,
        "Sehr stark": 1,
        "Sehr nützlich": 1,
        "Eher nützlich": 2,
        "Weniger nützlich": 3,
        "Überhaupt nicht nützlich": 4,
        "Gar nicht": 4,
        "Überhaupt nicht interessant": 4,
        "Überhaupt nicht wichtig": 4
    }

    # Anwenden der Likert-Skalen-Kodierung
    likert_columns = [
        "ar_familiarity", "vr_familiarity", "ar_vr_info_usefulness",
        "ar_vr_modern", "ar_vr_younger_target", "ar_vr_interest",
        "event_holistic"
    ]
    for col in likert_columns:
        df[col] = df[col].map(likert_map)

    # Kodierung von Ja/Nein/Unsicher-Fragen
    yes_no_map = {
        "Ja": 1,
        "Unsicher": 2,
        "Nein": 3
    }
    df["test_drive_intent_numeric"] = df["test_drive_intent"].map(yes_no_map)
    df["visit_bmw_open_more_numeric"] = df["visit_bmw_open_more"].map(yes_no_map)

    # Kodierung der AR/VR-Erfahrung
    df["ar_vr_experience_numeric"] = df["ar_vr_experience"].apply(
        lambda x: 1 if isinstance(x, str) and "Ja" in x else 0
    )

    # Konvertierung zu Integer
    df["brand_connection"] = df["brand_connection"].astype(int)
    df["brand_connection_after_ar_vr"] = df["brand_connection_after_ar_vr"].astype(int)

    return df

# Hilfsfunktion zur Speicherung der kodierten Daten
def save_encoded_data(df, file_path):
    """Speichert die kodierten Daten."""
    df.to_excel(file_path, index=False)
    print(f"Die kodierten Daten wurden gespeichert unter: {file_path}")


# Hilfsfunktion zur Definition von Teilgruppen
def define_subgroups(df):
    return {  # Gleiche Teilgruppen wie vorher
        "younger": lambda d: d[d["age_numeric"] <= 4],
        "older": lambda d: d[d["age_numeric"] > 4],
        "experienced": lambda d: d[d["ar_vr_experience_numeric"] == 1],
        "no_experience": lambda d: d[d["ar_vr_experience_numeric"] == 0],
        "male": lambda d: d[d["gender_numeric"] == 1],
        "female": lambda d: d[d["gender_numeric"] == 2],
    }

# Hilfsfunktion zur Berechnung der Spearman-Korrelation
def spearman_correlation_with_variance(df, xvar, yvar, subgroup_name, subgroup_func):
    subset = subgroup_func(df).dropna(subset=[xvar, yvar])
    n = len(subset)
    if n < 3:
        print(f"[{subgroup_name}] Nicht genug Daten für {xvar} vs {yvar} (n={n}).")
        return

    r, p = spearmanr(subset[xvar], subset[yvar])
    var_x = np.var(subset[xvar], ddof=1)
    var_y = np.var(subset[yvar], ddof=1)
    print(f"[{subgroup_name}] Korr {xvar} vs {yvar}: r={r:.3f}, p={p:.8f} (n={n})")
    print(f"[{subgroup_name}] Varianz {xvar}: {var_x:.3f}, Varianz {yvar}: {var_y:.3f}")

# Hilfsfunktion zur Berechnung des Chi-Quadrat-Tests
def calculate_chi2_and_print(df, question_variable, question_text):
    contingency_table = pd.crosstab(df['age_numeric'] <= 4, df[question_variable])
    contingency_table.index = ['Old', 'Young']
    print(f"\nKontingenztafel für: {question_text}\n", contingency_table)

    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print(f"\nChi-Quadrat Statistik: {chi2:.4f}")
    print(f"P-Wert: {p:.4f}")
    print(f"Freiheitsgrade: {dof}")

    alpha = 0.05
    if p < alpha:
        print(f"\nDer p-Wert ({p:.4f}) ist kleiner als das Signifikanzniveau ({alpha}).")
        print("=> Die Unterschiede in der Verteilung sind statistisch signifikant.")
        print("=> Nullhypothese (kein Zusammenhang) wird verworfen.")
    else:
        print(f"\nDer p-Wert ({p:.4f}) ist größer oder gleich dem Signifikanzniveau ({alpha}).")
        print("=> Es gibt keine statistisch signifikante Evidenz für Unterschiede.")
        print("=> Nullhypothese (kein Zusammenhang) kann nicht verworfen werden.")
    print("-" * 10, "\n")

# Hilfsfunktion zur Berechnung und Ausgabe der Verteilung
def calculate_and_print_distribution(df, column_name):
    distribution = df[column_name].value_counts(normalize=True) * 100
    print(f"\nVerteilung für '{column_name}':\n{distribution.sort_index()}")

# Hilfsfunktion zur Berechnung der AR/VR-Erfahrung nach Altersgruppen
def calculate_ar_vr_experience_by_age(df):
    young = df[df["age_numeric"] <= 4]
    old = df[df["age_numeric"] > 4]

    young_experience = young["ar_vr_experience"].value_counts(normalize=True).sort_index() * 100
    old_experience = old["ar_vr_experience"].value_counts(normalize=True).sort_index() * 100

    print("\nAR/VR Erfahrung - Junge Gruppe (<=34):\n", young_experience)
    print("\nAR/VR Erfahrung - Alte Gruppe (>34):\n", old_experience)

# Hilfsfunktion Korrelationsanalyse nach Untergruppen
def test_hypotheses(df, subgroups):
    # Teilhypothese 1
    print("\n=== Korrelationsanalyse ===")
    pairs_th1 = [
        ("ar_familiarity", "brand_connection_after_ar_vr"),
        ("ar_vr_interest", "brand_connection_after_ar_vr"),
        ("ar_excitement", "brand_connection_after_ar_vr"),
        ("vr_familiarity", "brand_connection_after_ar_vr"),
        ("vr_familiarity", "test_drive_intent_numeric"),
        ("ar_vr_interest", "test_drive_intent_numeric"),
        ("vr_emotion", "brand_connection_after_ar_vr"),
        ("ar_vr_interest", "event_holistic"),
        ("age_numeric", "ar_vr_interest"),
        ("age_numeric", "visit_bmw_open_more_numeric"),
        ("age_numeric", "event_holistic"),
        ("age_numeric", "ar_vr_younger_target")
    ]
    run_correlation_analysis(df, pairs_th1, subgroups)

# Hilfsfunktion zur Durchführung der Korrelationsanalyse
def run_correlation_analysis(df, pairs, subgroups):
    for xvar, yvar in pairs:
        df_sub = df.dropna(subset=[xvar, yvar])
        n_all = len(df_sub)
        if n_all > 2:
            r, p = spearmanr(df_sub[xvar], df_sub[yvar])
            print(f"[Gesamt] {xvar} vs {yvar}: r={r:.3f}, p={p:.8f} (n={n_all})")
        else:
            print(f"[Gesamt] Zu wenige Daten für {xvar} vs {yvar} (n={n_all}).")

        for subgroup_name, subgroup_func in subgroups.items():
            spearman_correlation_with_variance(df, xvar, yvar, subgroup_name, subgroup_func)
        print("----")


# Hauptfunktion
def main():
    df = load_and_rename_data(INPUT_FILE)
    df = encode_data(df)  # Daten kodieren
    subgroups = define_subgroups(df)  # Teilgruppen

    # Spearman-Korrelationen (Teilhypothesen)
    test_hypotheses(df, subgroups)

    # Chi-Quadrat-Tests (jetzt *ohne* Plot-Aufrufe)
    calculate_chi2_and_print(df, "ar_vr_interest", "Wie bewerten Sie die Idee, bei einem Event AR/VR-Erlebnisse zu erleben?")
    calculate_chi2_and_print(df, "ar_vr_younger_target", "Finden Sie, dass AR/VR-Technologien eher jüngere Zielgruppen ansprechen?")

    # Verteilungsberechnungen
    distribution_columns = [
        "ar_vr_younger_target", "ar_vr_interest", "test_drive_intent",
        "visit_bmw_open_more", "brand_connection_after_ar_vr", "brand_connection"
    ]
    for col in distribution_columns:
        calculate_and_print_distribution(df, col)

    # AR/VR-Erfahrung nach Alter
    calculate_ar_vr_experience_by_age(df)


if __name__ == "__main__":
    main()