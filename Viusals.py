import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns
from scipy.stats import spearmanr

matplotlib.use('TkAgg')
INPUT_FILE = r'Daten/AuswertungUmfrage.xlsx'

# Hilfsfunktion für die Annotation der Balken
def _annotate_bars(ax, rects, percent=False):
    for r in rects:
        height = r.get_height()
        label = f'{height:.1f}%' if percent else f'{height:.0f}'
        ax.annotate(label,
                    xy=(r.get_x() + r.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom')


# Hilfsfunktion für die Annotation der Pie-Chart
def _annotate_pie(ax, wedges, counts, total, offset=1.2):
    for i, wedge in enumerate(wedges):
        # Berechne den mittleren Winkel des Segments
        angle = (wedge.theta2 + wedge.theta1) / 2.0
        angle_rad = np.deg2rad(angle)
        # Bestimme den Startpunkt (am Kreisrand) und den Endpunkt (für die Textbox)
        x_start, y_start = np.cos(angle_rad), np.sin(angle_rad)
        x_end, y_end = offset * np.cos(angle_rad), offset * np.sin(angle_rad)
        # Berechne den Prozentwert (bezogen auf den Nenner total)
        pct = counts.iloc[i] / total * 100
        label = f"{pct:.1f}%"
        horizontalalignment = "left" if x_end >= 0 else "right"
        ax.annotate(
            label,
            xy=(x_start, y_start),
            xytext=(x_end, y_end),
            ha=horizontalalignment,
            va="center",
            textcoords="data",
            arrowprops=dict(
                arrowstyle="-",  # Einfache, gerade Linie
                color="black",
                lw=0.75,
                connectionstyle="arc3,rad=0"  # rad=0: völlig gerade Linie
            ),
            bbox=dict(
                boxstyle="round,pad=0.3",
                fc="white",
                ec="black",
                lw=0.5
            )
        )

# Spaltennamen kodieren um Übersichtlichkeit zu gewährleisten
def load_and_rename_data(file_path):
    df = pd.read_excel(file_path)
    mapping = {
        "Ihr Alter": "age", "Ihr Geschlecht": "gender",
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
    df.rename(columns=mapping, inplace=True)
    df["age_numeric"] = df["age"].map({"<18": 1, "18-24": 2, "25-34": 3, "35-44": 4, "45-54": 5, ">55": 6})
    df["gender_numeric"] = df["gender"].map({"männlich": 1, "weiblich": 2, "divers": 3})
    return df

# Hilfsfunktion zur Darstellung der Spearman-Korrelation
def plot_spearman_correlation(df, title="Spearman Correlation Heatmap", columns=None):
    if columns:
        df = df[columns]
    num_df = df.select_dtypes(include=[np.number])
    corr, _ = spearmanr(num_df, axis=0, nan_policy='omit')
    corr_df = pd.DataFrame(corr, index=num_df.columns, columns=num_df.columns)
    mask = np.eye(corr_df.shape[0], dtype=bool)
    corr_df_masked = corr_df.mask(mask, other=np.nan)
    plt.figure(figsize=(12, 10))
    ax = sns.heatmap(corr_df_masked, annot=True, cmap="coolwarm", fmt=".2f",
                     linewidths=0.5, vmin=-1, vmax=1)
    for i in range(corr_df.shape[0]):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=True, color='black', lw=0))
    plt.subplots_adjust(left=0.2, bottom=0.4, right=0.95, top=0.95)
    plt.title(title)
    plt.show()

# Hilfsfunktion zur Darstellung des Vergleichs von Altersgruppen
def plot_age_group_comparison(df, var, xlabel, order=None):
    young = df[df["age_numeric"] <= 4]
    old = df[df["age_numeric"] > 4]
    if order:
        yc = young[var].value_counts(normalize=True).reindex(order).fillna(0) * 100
        oc = old[var].value_counts(normalize=True).reindex(order).fillna(0) * 100
    else:
        yc = young[var].value_counts(normalize=True).sort_index() * 100
        oc = old[var].value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(yc))
    r1 = ax.bar(x - 0.35 / 2, yc, 0.35, label="Young (<=44)", color='orange')
    r2 = ax.bar(x + 0.35 / 2, oc, 0.35, label="Old (>44)", color='blue')
    _annotate_bars(ax, r1, percent=True)
    _annotate_bars(ax, r2, percent=True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Anteil in Prozent")
    ax.set_title('Vergleich der Antwortverteilung nach Altersgruppen')
    ax.set_xticks(x)
    ax.set_xticklabels(order if order else yc.index, rotation=45, ha="right")
    ax.legend()
    plt.ylim(0, 70)
    plt.tight_layout()
    plt.show()

# Hilfsfunktion zur Darstellung des Vergleichs von Brand Connection
def plot_brand_connection_comparison(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(5)
    bc = df["brand_connection"].value_counts(normalize=True).sort_index() * 100
    bac = df["brand_connection_after_ar_vr"].value_counts(normalize=True).sort_index() * 100
    r1 = ax.bar(x - 0.35 / 2, bc, 0.35, label="Brand Connection", color='blue')
    r2 = ax.bar(x + 0.35 / 2, bac, 0.35, label="Brand Connection after AR/VR", color='orange')
    _annotate_bars(ax, r1, percent=True)
    _annotate_bars(ax, r2, percent=True)
    ax.set_xlabel("Brand Connection Level")
    ax.set_ylabel("Anteil in Prozent")
    ax.set_title('Comparison of Brand Connection Before and After AR/VR')
    ax.set_xticks(x)
    ax.set_xticklabels(bc.index, rotation=45, ha="right")
    ax.legend()
    plt.ylim(0, 50)
    plt.tight_layout()
    plt.show()


# Hilfsfunktion zur Darstellung des Kreisdiagramms
def plot_pie_chart(df, col, title=None):
    counts = df[col].value_counts()
    total = df[col].notna().sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('Set3', len(counts))
    wedges, _ = ax.pie(counts, startangle=140, colors=cmap(range(len(counts))),
                       wedgeprops=dict(edgecolor='w'))
    ax.legend(wedges, counts.index, title="Antworten", loc="center left", bbox_to_anchor=(1.1, 0.5))

    # Nutze die globale _annotate_pie-Funktion für die Annotation
    _annotate_pie(ax, wedges, counts, total, offset=1.2)

    ax.set_title(title if title else f'Verteilung der Antworten für "{col}"', fontsize=16)
    plt.tight_layout()
    plt.show()

# Hilfsfunktion zur Darstellung des Kreisdiagramms für Mehrfachauswahl
def plot_pie_chart_multiselect_I(df, col, title=None):
    # Splitte den String an Kommas (mit einem Lookbehind, falls ein Punkt vorhanden ist)
    s = df[col].str.split(r'(?<=\.)\,').explode().str.strip()
    s = s[s != ""]  # Entferne leere Ergebnisse

    counts = s.value_counts()
    total = df[col].notna().sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('Set3', len(counts))
    wedges, _ = ax.pie(counts, startangle=140, colors=cmap(range(len(counts))),
                       wedgeprops=dict(edgecolor='w'))
    ax.legend(wedges, counts.index, title="Antworten", loc="center left", bbox_to_anchor=(1.1, 0.5))

    # Nutze die _annotate_pie-Funktion
    _annotate_pie(ax, wedges, counts, total, offset=1.2)

    ax.set_title(title if title else f'Verteilung der Antworten für "{col}"', fontsize=16)
    ax.set_position([0.05, 0.1, 0.5, 0.8])
    plt.show()


# Hilfsfunktion zur Darstellung des Kreisdiagramms für Mehrfachauswahl
def plot_pie_chart_multiselect_II(df, col, title=None, sep=','):
    def custom_split(x):
        if pd.isna(x):
            return []
        x = x.strip()
        # Sonderfälle vereinheitlichen
        if x in ["Diese Frage trifft nicht zu, da ich die BMW Open noch nicht besucht habe.", "/"]:
            return ["Diese Frage trifft nicht zu, da ich die BMW Open noch nicht besucht habe."]
        return list({i.strip() for i in x.split(sep) if i.strip()})

    s = df[col].apply(custom_split).explode().loc[lambda s: s != ""].dropna()
    counts = s.value_counts()
    total = df[col].notna().sum()

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('Set3', len(counts))
    wedges, _ = ax.pie(counts, startangle=140, colors=cmap(range(len(counts))),
                       wedgeprops=dict(edgecolor='w'))
    ax.legend(wedges, counts.index, title="Antworten", loc="center left", bbox_to_anchor=(1.1, 0.5))

    _annotate_pie(ax, wedges, counts, total, offset=1.2)

    ax.set_title(title if title else f'Verteilung der Antworten für "{col}"', fontsize=16)
    ax.set_position([0.05, 0.1, 0.5, 0.8])
    plt.show()


# Hilfsspalten für die Darstellung des Kreisdiagramms für die Gründe des BMW Open-Besuchs
def plot_pie_chart_why_visited_bmw_open(df, col="why_visited_bmw_open", title=None, offset=1.2):
    rec = [
        "Diese Frage trifft nicht zu, da ich die BMW Open noch nicht besucht habe.",
        "Noch nie besucht",
        "Interesse am Tennissport",
        "Interesse an Sportveranstaltungen",
        "Neugier auf das Event und die Atmosphäre",
        "Begeisterung für die Marke BMW",
        "Einladung durch Freunde, Familie oder Geschäftspartner",
        "Interesse an speziellen Aktionen oder Attraktionen vor Ort",
        "Networking-Möglichkeiten",
        "Tradition oder regelmäßiger Besuch von Sportveranstaltungen"
    ]

    def custom_extract(x):
        if pd.isna(x):
            return []
        x = x.strip()
        if x in ["Noch nie besucht", "/", ""]:
            return [rec[0]]
        return [r for r in rec if r in x]

    total_respondents = df[col].notna().sum()
    counts_dict = {r: 0 for r in rec}
    for x in df[col]:
        for answer in set(custom_extract(x)):
            if answer in counts_dict:
                counts_dict[answer] += 1

    counts = pd.Series([counts_dict[r] for r in rec], index=rec)
    counts = counts[counts > 0]

    if total_respondents == 0:
        print("Hinweis: Es wurden keine Antworten zu 'why_visited_bmw_open' gefunden.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    cmap = plt.cm.get_cmap('Set3', len(counts))
    wedges, _ = ax.pie(counts, startangle=140, colors=cmap(range(len(counts))),
                       wedgeprops=dict(edgecolor='w'))
    ax.legend(wedges, counts.index, title="Antworten", loc="center left", bbox_to_anchor=(1.1, 0.5))

    _annotate_pie(ax, wedges, counts, total_respondents, offset=offset)

    ax.set_title(title if title else 'Gründe für den bisherigen BMW Open Besuch', fontsize=16)
    plt.tight_layout()
    plt.show()


# Hilfsfunktion zur Darstellung der Balkendiagramme
def plot_bar_chart(df, col, title=None, order=None):
    counts = (df[col].value_counts().reindex(order).fillna(0)
              if order else df[col].value_counts().sort_index())
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(counts.index.astype(str), counts.values, color="#0000ff")
    _annotate_bars(ax, bars, percent=False)
    ax.set_title(title if title else f'Verteilung der Antworten für "{col}"', fontsize=16)
    ax.set_xlabel(col)
    ax.set_ylabel('Anzahl')
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()

# Hilfsfunktion zur Kodierung der Daten
def encode_data(df):
    likert = {
        "Ja, auf jeden Fall": 1, "Ja, teilweise": 2, "Nein": 3,
        "Gar nicht vertraut": 4, "Kaum vertraut": 3, "Etwas vertraut": 2, "Sehr vertraut": 1,
        "Sehr uninteressant": 1, "Weniger interessant": 3, "Neutral": 3,
        "Eher interessant": 2, "Sehr interessant": 1,
        "Weniger wichtig": 3, "Eher wichtig": 2, "Sehr wichtig": 1,
        "Weniger stark": 3, "Eher stark": 2, "Sehr stark": 1,
        "Sehr nützlich": 1, "Eher nützlich": 2, "Weniger nützlich": 3,
        "Überhaupt nicht nützlich": 4, "Gar nicht": 4,
        "Überhaupt nicht interessant": 4, "Überhaupt nicht wichtig": 4
    }
    for col in ["ar_familiarity", "vr_familiarity", "ar_vr_info_usefulness",
                "ar_vr_modern", "ar_vr_younger_target", "ar_vr_interest", "event_holistic"]:
        if col in df:
            df[col] = df[col].map(likert)
    for orig, new in [("test_drive_intent", "test_drive_intent_numeric"),
                      ("visit_bmw_open_more", "visit_bmw_open_more_numeric")]:
        if orig in df:
            df[new] = df[orig].map({"Ja": 1, "Unsicher": 2, "Nein": 3})
    if "ar_vr_experience" in df:
        df["ar_vr_experience_numeric"] = df["ar_vr_experience"].apply(lambda x: 1 if isinstance(x, str) and "Ja" in x else 0)
    for col in ["brand_connection", "brand_connection_after_ar_vr", "bmw_open_frequency", "ar_excitement", "vr_emotion"]:
        if col in df:
            df[col] = df[col].astype(int)
    return df

# Hilfsfunktion Subgruppen
def define_subgroups(df):
    return {
        "younger": lambda d: d[d["age_numeric"] <= 4],
        "older": lambda d: d[d["age_numeric"] > 4],
        "experienced": lambda d: d[d["ar_vr_experience_numeric"] == 1],
        "no_experience": lambda d: d[d["ar_vr_experience_numeric"] == 0],
        "male": lambda d: d[d["gender_numeric"] == 1],
        "female": lambda d: d[d["gender_numeric"] == 2],
    }

# Hauptfunktion
def main():
    df = load_and_rename_data(INPUT_FILE)
    df_encoded = encode_data(df.copy())
    subgroups = define_subgroups(df_encoded)
    corr_pairs = [("ar_familiarity", "brand_connection_after_ar_vr"),
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
                  ("age_numeric", "ar_vr_younger_target"),
                  ("bmw_open_frequency", "brand_connection")]
    heatmap_cols = list({c for pair in corr_pairs for c in pair if c in df_encoded and pd.api.types.is_numeric_dtype(df_encoded[c])})
    plot_spearman_correlation(df_encoded, "Gesamte Spearman-Korrelation", columns=heatmap_cols)
    for name, func in subgroups.items():
        plot_spearman_correlation(func(df_encoded), f"Spearman-Korrelation ({name})", columns=heatmap_cols)
    order_int = ["Sehr interessant", "Eher interessant", "Weniger interessant", "Überhaupt nicht interessant"]
    if "ar_vr_interest" in df:
        plot_age_group_comparison(df, "ar_vr_interest",
                                  "ar_vr_interest",
                                  order_int)
    order_younger = ["Ja, auf jeden Fall", "Ja, teilweise", "Nein"]
    if "ar_vr_younger_target" in df:
        plot_age_group_comparison(df, "ar_vr_younger_target",
                                  "ar_vr_younger_target",
                                  order_younger)
    if "brand_connection" in df and "brand_connection_after_ar_vr" in df:
        plot_brand_connection_comparison(df)

    plot_pie_chart_why_visited_bmw_open(df)
    plot_pie_chart_multiselect_I(df, "why_never_visited", "Gründe warum die BMW Open noch nie besucht wurden")
    plot_pie_chart_multiselect_II(df, "ar_vr_which_experiences", "Bevorzugte AR/VR-Erlebnisse bei einem BMW-Event")
    plot_pie_chart(df, "age", "Alter der Befragten")
    plot_pie_chart(df, "gender", "Geschlecht der Befragten")
    plot_pie_chart(df, "employment_status", "Beschäftigungsstatus der Befragten")
    plot_pie_chart(df, "ar_vr_experience", "Erfahrung mit AR/VR-Technologien")
    plot_pie_chart(df, "ar_familiarity", "Vertrautheit mit AR-Technologie")
    plot_pie_chart(df, "vr_familiarity", "Vertrautheit mit VR-Technologie")
    plot_pie_chart(df, "test_drive_intent", "Bereitschaft zur Vereinbarung einer Testfahrt")
    plot_pie_chart(df, "event_holistic", "Wichtigkeit eines ganzheitlichen Event-Erlebnisses")
    plot_pie_chart(df, "ar_vr_younger_target", "Ansprechbarkeit jüngerer Zielgruppen durch AR/VR")
    plot_pie_chart(df, "ar_vr_modern", "Darstellung der Marke BMW als modern und fortschrittlich durch AR/VR")
    plot_pie_chart(df, "visit_bmw_open_more", "Häufigere Besuche der BMW Open durch positive AR/VR-Erfahrungen")
    plot_bar_chart(df_encoded, "ar_vr_interest", "Bewertung der Idee, AR/VR-Erlebnisse bei einem Event zu erleben")
    plot_bar_chart(df_encoded, "ar_vr_info_usefulness", "Bewertung der AR/VR-Technologien zur Informationsgewinnung")
    plot_bar_chart(df, "brand_connection", "Wie stark fühlen Sie sich mit der Marke BMW verbunden?")
    plot_bar_chart(df, "bmw_open_frequency", "Wie oft besuchen Sie die BMW Open?")
    plot_bar_chart(df, "ar_excitement", "Mittels AR digitale Inhalte sehen und interagieren")
    plot_bar_chart(df, "vr_emotion", "Emotionen bei virtuellem Rennen auf dem Nürburgring")
    plot_bar_chart(df, "brand_connection_after_ar_vr", "Brand Connection nach AR/VR-Erfahrung")


if __name__ == "__main__":
    main()
