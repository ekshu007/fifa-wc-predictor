import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import xgboost as xgb

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="FIFA World Cup Predictor",
    page_icon="🏆",
    layout="wide"
)

st.markdown("""
    <style>
    .main { background-color: #0e1117; }
    h1 { color: #f5a623; font-family: 'Georgia', serif; }
    h2, h3 { color: #e0e0e0; }
    .stMetric { background-color: #1c1f26; border-radius: 10px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────
# LOAD & CACHE DATA
# ─────────────────────────────────────────
@st.cache_data
def load_and_process():
    matches  = pd.read_csv("../data/WorldCupMatches.csv")
    rankings = pd.read_csv("../data/fifa_ranking-2024-06-20.csv")
    players  = pd.read_csv("../data/WorldCupPlayers.csv")

    # Clean rankings
    rankings["rank_date"] = pd.to_datetime(rankings["rank_date"])
    rankings = rankings.sort_values(["country_full", "rank_date"]).reset_index(drop=True)

    # Clean matches
    matches["Home Team Name"] = matches["Home Team Name"].str.replace(r'.*rn">', '', regex=True).str.strip()
    matches["Away Team Name"] = matches["Away Team Name"].str.replace(r'.*rn">', '', regex=True).str.strip()
    matches["Home Team Name"] = matches["Home Team Name"].apply(lambda x: "Côte d'Ivoire" if isinstance(x, str) and "voire" in x else x)
    matches["Away Team Name"] = matches["Away Team Name"].apply(lambda x: "Côte d'Ivoire" if isinstance(x, str) and "voire" in x else x)

    name_mapping = {
        "Serbia and Montenegro": "Serbia",
        "Czech Republic": "Czechia",
        "Iran": "IR Iran",
        "Ireland": "Republic of Ireland",
    }
    matches["Home Team Name"] = matches["Home Team Name"].replace(name_mapping)
    matches["Away Team Name"] = matches["Away Team Name"].replace(name_mapping)
    matches = matches[matches["Year"] >= 1994].copy()
    matches["Year"] = matches["Year"].astype(int)

    # Build initials → full name map from matches
    home_map = matches[["Home Team Name", "Home Team Initials"]].rename(
        columns={"Home Team Name": "team", "Home Team Initials": "initials"})
    away_map = matches[["Away Team Name", "Away Team Initials"]].rename(
        columns={"Away Team Name": "team", "Away Team Initials": "initials"})
    initials_map = pd.concat([home_map, away_map]).drop_duplicates()

    # Add year to players via matches
    players = players.merge(
        matches[["MatchID", "Year"]].drop_duplicates(), on="MatchID", how="left"
    )
    players = players.merge(initials_map, left_on="Team Initials", right_on="initials", how="left")
    players = players[players["Year"] >= 1994].copy()

    return matches, rankings, players

@st.cache_data
def build_features(matches, rankings):
    tournament_dates = {
        1994: "1994-06-17", 1998: "1998-06-10", 2002: "2002-05-31",
        2006: "2006-06-09", 2010: "2010-06-11", 2014: "2014-06-12",
        2018: "2018-06-14", 2022: "2022-11-20"
    }

    def get_ranking_at_tournament(team, year):
        cutoff = pd.to_datetime(tournament_dates[year])
        team_ranks = rankings[(rankings["country_full"] == team) & (rankings["rank_date"] <= cutoff)]
        if team_ranks.empty:
            return None
        return team_ranks.sort_values("rank_date").iloc[-1]["rank"]

    team_years = pd.concat([
        matches[["Year", "Home Team Name"]].rename(columns={"Home Team Name": "team"}),
        matches[["Year", "Away Team Name"]].rename(columns={"Away Team Name": "team"})
    ]).drop_duplicates()
    team_years["fifa_ranking"] = team_years.apply(lambda r: get_ranking_at_tournament(r["team"], r["Year"]), axis=1)

    def map_stage(stage):
        if stage == "Final": return 5
        elif stage == "Semi-finals": return 4
        elif stage == "Quarter-finals": return 3
        elif stage == "Round of 16": return 2
        elif stage in ["Match for third place", "Third place", "Play-off for third place"]: return 4
        elif "Group" in str(stage): return 1
        return 1

    known_winners = {2018: "France", 2022: "Argentina"}

    def get_winner(matches, year):
        if year in known_winners:
            return known_winners[year]
        final = matches[(matches["Year"] == year) & (matches["Stage"] == "Final")]
        if final.empty:
            return None
        row = final.iloc[0]
        if row["Home Team Goals"] > row["Away Team Goals"]: return row["Home Team Name"]
        elif row["Away Team Goals"] > row["Home Team Goals"]: return row["Away Team Name"]
        else:
            wc = str(row["Win conditions"])
            if "penalties" in wc.lower():
                return row["Home Team Name"] if row["Home Team Name"].split()[0] in wc else row["Away Team Name"]
        return None

    def build_team_features(matches, year):
        rows = []
        teams = pd.unique(matches[["Home Team Name", "Away Team Name"]].values.ravel())
        winner = get_winner(matches, year)
        for team in teams:
            home = matches[matches["Home Team Name"] == team]
            away = matches[matches["Away Team Name"] == team]
            goals_scored   = home["Home Team Goals"].sum() + away["Away Team Goals"].sum()
            goals_conceded = home["Away Team Goals"].sum() + away["Home Team Goals"].sum()
            total_matches  = len(home) + len(away)
            wins = len(home[home["Home Team Goals"] > home["Away Team Goals"]]) + \
                   len(away[away["Away Team Goals"] > away["Home Team Goals"]])
            all_stages = pd.concat([home["Stage"], away["Stage"]])
            best_stage = all_stages.apply(map_stage).max()
            ranking_row = team_years[(team_years["team"] == team) & (team_years["Year"] == year)]
            ranking = ranking_row["fifa_ranking"].values[0] if len(ranking_row) > 0 else None
            rows.append({
                "team": team, "year": year,
                "goals_scored": goals_scored, "goals_conceded": goals_conceded,
                "goal_diff": goals_scored - goals_conceded,
                "goals_per_game": round(goals_scored / total_matches, 2),
                "wins": wins, "total_matches": total_matches,
                "best_stage": best_stage, "fifa_ranking": ranking,
                "won_tournament": 1 if team == winner else 0
            })
        return pd.DataFrame(rows)

    all_years = [1994, 1998, 2002, 2006, 2010, 2014]
    team_features = pd.concat([build_team_features(matches[matches["Year"] == y], y) for y in all_years], ignore_index=True)
    return team_features

@st.cache_resource
def train_model(team_features):
    features = ["goals_scored", "goals_conceded", "goal_diff",
                "goals_per_game", "wins", "total_matches", "best_stage", "fifa_ranking"]
    train = team_features[team_features["year"] <= 2010]
    X_train, y_train = train[features], train["won_tournament"]
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    lr = LogisticRegression(class_weight="balanced", max_iter=1000)
    lr.fit(X_train_sc, y_train)
    return lr, scaler, features

@st.cache_data
def simulate_2026(team_features, rankings):
    qualified_2026 = [
        "Argentina", "France", "Belgium", "Brazil", "England", "Portugal",
        "Netherlands", "Spain", "Croatia", "Italy", "USA", "Colombia",
        "Morocco", "Uruguay", "Mexico", "Germany", "Japan", "Senegal",
        "Switzerland", "IR Iran", "Australia", "Korea Republic", "Poland",
        "Denmark", "Austria", "Ecuador", "Peru", "Chile", "Venezuela",
        "Canada", "Costa Rica", "Panama", "Honduras", "Jamaica",
        "Nigeria", "Cameroon", "Ghana", "Egypt", "Algeria", "Tunisia",
        "South Africa", "Mali", "Saudi Arabia", "Qatar", "Iraq",
        "Uzbekistan", "New Zealand", "Slovakia"
    ]
    latest_date = rankings["rank_date"].max()
    latest_rankings = rankings[rankings["rank_date"] == latest_date][["country_full", "rank"]]
    teams_2026 = pd.DataFrame({"team": qualified_2026})
    teams_2026 = teams_2026.merge(latest_rankings, left_on="team", right_on="country_full", how="left").drop(columns="country_full")
    teams_2026["fifa_ranking"] = teams_2026["rank"]

    stat_features = ["goals_scored", "goals_conceded", "goal_diff", "goals_per_game", "wins", "total_matches", "best_stage"]
    team_features_clean = team_features.dropna(subset=["fifa_ranking"])
    for stat in stat_features:
        reg = LinearRegression()
        reg.fit(team_features_clean[["fifa_ranking"]], team_features_clean[stat])
        teams_2026[stat] = reg.predict(teams_2026[["fifa_ranking"]])

    return teams_2026

# ─────────────────────────────────────────
# MAIN APP
# ─────────────────────────────────────────
st.title("🏆 FIFA World Cup Predictor")
st.caption("Historical tournament explorer + 2026 win probability predictions")

with st.spinner("Loading data and training model..."):
    matches, rankings, players = load_and_process()
    team_features = build_features(matches, rankings)
    lr, scaler, features = train_model(team_features)
    teams_2026 = simulate_2026(team_features, rankings)

    X_2026 = teams_2026[features]
    X_2026_sc = scaler.transform(X_2026)
    raw_probs = lr.predict_proba(X_2026_sc)[:, 1]
    teams_2026["win_probability"] = raw_probs / raw_probs.sum()
    teams_2026 = teams_2026.sort_values("win_probability", ascending=False).reset_index(drop=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🏆 2026 Predictions", "📊 Tournament Explorer", "🔍 Team Deep Dive"])

# ── TAB 1: 2026 Predictions ──────────────
with tab1:
    st.subheader("2026 FIFA World Cup — Predicted Win Probabilities")
    top_n = st.slider("Show top N teams", 5, 48, 15)
    top = teams_2026.head(top_n)

    fig, ax = plt.subplots(figsize=(10, top_n * 0.45 + 1))
    colors = ["#f5a623" if i == 0 else "#4a90d9" for i in range(len(top))]
    ax.barh(top["team"][::-1], top["win_probability"][::-1], color=colors[::-1])
    ax.set_xlabel("Win Probability")
    ax.set_title(f"Top {top_n} Predicted Teams for 2026")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        top[["team", "rank", "win_probability"]].rename(columns={"team": "Team", "rank": "FIFA Rank", "win_probability": "Win Probability"}).reset_index(drop=True),
        use_container_width=True
    )

# ── TAB 2: Tournament Explorer ───────────
with tab2:
    st.subheader("Historical Tournament Explorer")
    year = st.selectbox("Select Tournament Year", [1994, 1998, 2002, 2006, 2010, 2014])
    year_data = team_features[team_features["year"] == year].sort_values("best_stage", ascending=False)

    col1, col2, col3, col4, col5 = st.columns(5)
    winner_row = year_data[year_data["won_tournament"] == 1]
    if not winner_row.empty:
        col1.metric("🥇 Winner", winner_row.iloc[0]["team"])
        col2.metric("Winner Goals", f"{winner_row.iloc[0]['goals_scored']:.0f}")
        col3.metric("Winner FIFA Rank", f"{winner_row.iloc[0]['fifa_ranking']:.0f}")
    col4.metric("⚽ Total Goals", f"{year_data['goals_scored'].sum():.0f}")
    col5.metric("🏳️ Teams", f"{len(year_data)}")

    st.dataframe(
        year_data[["team", "fifa_ranking", "goals_scored", "goals_conceded", "goal_diff", "goals_per_game", "wins", "best_stage", "won_tournament"]]
        .rename(columns={
            "team": "Team", "fifa_ranking": "FIFA Rank", "goals_scored": "Goals Scored",
            "goals_conceded": "Goals Conceded", "goal_diff": "Goal Diff",
            "goals_per_game": "Goals/Game", "wins": "Wins",
            "best_stage": "Best Stage", "won_tournament": "Winner"
        }).reset_index(drop=True),
        use_container_width=True
    )

# ── TAB 3: Team Deep Dive ────────────────
with tab3:
    st.subheader("Team Deep Dive")
    all_teams = sorted(team_features["team"].unique().tolist())
    selected_team = st.selectbox("Select a Team", all_teams)

    team_data = team_features[team_features["team"] == selected_team].sort_values("year")

    if team_data.empty:
        st.warning("No data found for this team.")
    else:
        st.write(f"**Tournaments participated:** {len(team_data)}")

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(team_data["year"], team_data["goals_per_game"], marker="o", color="#f5a623", linewidth=2)
            ax.set_title("Goals Per Game by Year")
            ax.set_xlabel("Year")
            ax.set_ylabel("Goals/Game")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(team_data["year"], team_data["fifa_ranking"], marker="o", color="#4a90d9", linewidth=2)
            ax.invert_yaxis()
            ax.set_title("FIFA Ranking by Year (lower = better)")
            ax.set_xlabel("Year")
            ax.set_ylabel("FIFA Rank")
            ax.spines[["top", "right"]].set_visible(False)
            plt.tight_layout()
            st.pyplot(fig)

        st.dataframe(
            team_data[["year", "fifa_ranking", "goals_scored", "goals_conceded", "goal_diff", "wins", "best_stage", "won_tournament"]]
            .rename(columns={
                "year": "Year", "fifa_ranking": "FIFA Rank", "goals_scored": "Goals Scored",
                "goals_conceded": "Goals Conceded", "goal_diff": "Goal Diff",
                "wins": "Wins", "best_stage": "Best Stage", "won_tournament": "Winner"
            }).reset_index(drop=True),
            use_container_width=True
        )

        # 2026 prediction for selected team
        team_2026 = teams_2026[teams_2026["team"] == selected_team]
        if not team_2026.empty:
            prob = team_2026.iloc[0]["win_probability"]
            rank_2026 = int(teams_2026[teams_2026["team"] == selected_team].index[0]) + 1
            st.markdown("---")
            st.metric(f"🔮 2026 Win Probability", f"{prob:.1%}", f"Ranked #{rank_2026} among 2026 teams")

        # ── PLAYERS SECTION ──
        st.markdown("---")
        st.subheader("👕 Squad Data")

        team_players = players[players["team"] == selected_team].copy()

        if team_players.empty:
            st.info("No player data available for this team.")
        else:
            # Year filter
            available_years = sorted(team_players["Year"].dropna().unique().astype(int).tolist())
            selected_year = st.selectbox("Select Tournament Year", available_years, key="player_year")
            year_players = team_players[team_players["Year"] == selected_year]

            col1, col2, col3 = st.columns(3)
            col1.metric("Total Players", len(year_players["Player Name"].unique()))
            col2.metric("Coach", year_players["Coach Name"].iloc[0] if not year_players.empty else "N/A")

            # Goals from Event column (G followed by minute)
            goals = year_players[year_players["Event"].str.contains(r"G\d", na=False)]
            col3.metric("Goal Events", len(goals))

            # Tabs within team deep dive
            p_tab1, p_tab2, p_tab3 = st.tabs(["📋 Full Squad", "⚽ Goalscorers", "🟥 Cards & Events"])

            with p_tab1:
                squad = year_players[["Shirt Number", "Player Name", "Position", "Line-up"]].drop_duplicates("Player Name")
                squad = squad.rename(columns={"Shirt Number": "#", "Line-up": "Start/Sub"})
                squad["Start/Sub"] = squad["Start/Sub"].map({"S": "Starting", "N": "Substitute"})
                st.dataframe(squad.sort_values("#").reset_index(drop=True), use_container_width=True)

            with p_tab2:
                goal_rows = year_players[year_players["Event"].str.contains(r"G\d", na=False)][["Player Name", "Event"]].copy()
                # Extract all goal minutes from event string
                goal_rows["Goals"] = goal_rows["Event"].str.findall(r"G(\d+)'").apply(len)
                goal_summary = goal_rows.groupby("Player Name")["Goals"].sum().reset_index().sort_values("Goals", ascending=False)
                if goal_summary.empty:
                    st.info("No goal data found.")
                else:
                    st.dataframe(goal_summary.reset_index(drop=True), use_container_width=True)

                    fig, ax = plt.subplots(figsize=(8, max(3, len(goal_summary) * 0.4)))
                    ax.barh(goal_summary["Player Name"][::-1], goal_summary["Goals"][::-1], color="#f5a623")
                    ax.set_xlabel("Goals")
                    ax.set_title(f"{selected_team} — Top Scorers {selected_year}")
                    ax.spines[["top", "right"]].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)

            with p_tab3:
                # Red cards (R), Yellow (Y), Substitutions (W = withdrawn, I = in)
                event_rows = year_players[year_players["Event"].notna()][["Player Name", "Event"]].copy()
                event_rows["Red Card"]    = event_rows["Event"].str.contains(r"R\d", na=False)
                event_rows["Yellow Card"] = event_rows["Event"].str.contains(r"Y\d", na=False)
                event_rows["Subbed Off"]  = event_rows["Event"].str.contains(r"W\d", na=False)
                event_rows["Subbed On"]   = event_rows["Event"].str.contains(r"I\d", na=False)
                event_rows = event_rows[event_rows[["Red Card", "Yellow Card", "Subbed Off", "Subbed On"]].any(axis=1)]
                if event_rows.empty:
                    st.info("No card/substitution data found.")
                else:
                    st.dataframe(event_rows[["Player Name", "Event", "Red Card", "Yellow Card", "Subbed Off", "Subbed On"]].reset_index(drop=True), use_container_width=True)
