**🏆 FIFA World Cup Winner Predictor**
A machine learning project that predicts FIFA World Cup winners using historical match data and FIFA rankings. Built with Python, Scikit-learn, XGBoost, and Streamlit.
**
📌 Project Overview**
This project answers one question: can we predict which team will win the FIFA World Cup based on historical performance and FIFA rankings?
We train a classification model on World Cup data from 1994–2014, evaluate it on the 2014 tournament, and then simulate win probabilities for all 48 teams in the 2026 FIFA World Cup.
The project also includes an interactive Streamlit dashboard to explore historical tournaments, team stats, player data, and 2026 predictions.

**🗂️ Project Structure**
fifa-wc-predictor/
├── data/                          # Raw CSV datasets (not tracked by git)
│   ├── WorldCupMatches.csv        # Match results for every WC (1930–2014)
│   ├── WorldCupPlayers.csv        # Player-level data (goals, cards, subs)
│   └── fifa_ranking-2024-06-20.csv  # FIFA rankings from 1992 to 2024
├── model/
│   ├── fifa_wc_predictor.ipynb    # Full analysis notebook
│   └── app.py                     # Streamlit dashboard
├── .gitignore
└── README.md

**⚙️ How It Works**
1. Data Collection
Three datasets are used:

WorldCupMatches.csv — real match results including scores, stages, and teams for every World Cup
WorldCupPlayers.csv — player-level data including goals scored, yellow/red cards, and substitutions
FIFA Rankings CSV — official FIFA team rankings published monthly since 1992

**2. Data Cleaning**

HTML encoding artifacts in team names are stripped
Team names are standardized across datasets (e.g. "IR Iran" vs "Iran", "Korea Republic" vs "South Korea")
Rankings are matched to each team at the exact date before each tournament to avoid data leakage

**3. Feature Engineering**
For each team in each tournament, we engineer the following features:
FeatureDescriptiongoals_scoredTotal goals scored in the tournamentgoals_concededTotal goals concededgoal_diffGoal differencegoals_per_gameAverage goals scored per matchwinsNumber of wins in 90 minutestotal_matchesTotal matches playedbest_stageFurthest stage reached (Group=1 → Final=5)fifa_rankingFIFA ranking at tournament start
This produces one row per team per tournament — 184 rows across 6 tournaments (1994–2014).
4. Model Training
Three models are trained and compared:

Logistic Regression — with class balancing to handle the imbalanced target (only 1 winner per tournament)
Random Forest — ensemble tree model with class weights
XGBoost — gradient boosting with scale_pos_weight for imbalance

Training set: 1994–2010 tournaments
Test set: 2014 tournament
Logistic Regression is selected as the final model due to its ability to correctly identify the winner (high recall) while remaining interpretable.
5. 2026 Simulation
Since 2026 match data doesn't exist yet, we simulate it:

Get the latest FIFA ranking for all 48 qualified teams
Use a linear regression trained on historical data to estimate each team's expected stats (goals, wins, stage) based on their ranking
Feed those estimated stats into the trained Logistic Regression model
Normalize raw probabilities so all 48 teams sum to 100%
Rank teams by predicted win probability


**📊 Streamlit Dashboard**
The dashboard has 3 tabs:
🏆 2026 Predictions

Bar chart of predicted win probabilities for all 48 qualified teams
Adjustable slider to show top N teams
Full probability table

**📊 Tournament Explorer**

Select any tournament year (1994–2014)
See all teams ranked by performance
Summary metrics: winner, total goals, number of teams

**🔍 Team Deep Dive**

Select any team to see their stats across all tournaments
Goals per game and FIFA ranking trend charts
Squad data — full player list, goalscorers, cards and substitutions per tournament
2026 win probability for the selected team


**🚀 Getting Started**
Prerequisites
bashpip install pandas scikit-learn xgboost streamlit matplotlib
Data
Download the following datasets from Kaggle and place them in the /data folder:

FIFA World Cup Dataset
FIFA World Rankings

Run the Notebook
Open model/fifa_wc_predictor.ipynb in Jupyter and run all cells.
Run the Dashboard
bashcd model
streamlit run app.py

📈 Results
The Logistic Regression model correctly identifies Germany as the most likely winner of the 2014 World Cup (who did win), with Netherlands and Argentina close behind — matching the actual finalists and semi-finalists.
For 2026, the model favors highly ranked teams like Argentina, France, Brazil and England based on their current FIFA rankings and historical tournament performance patterns.

**⚠️ Limitations**

Training data covers only 6 tournaments (1994–2014) due to dataset availability
2026 stats are estimated from rankings, not real match data
Model does not account for injuries, squad depth, manager quality, or tournament draws
FIFA rankings alone don't fully capture team strength at any given moment


**🛠️ Built With**

Python
Pandas
Scikit-learn
XGBoost
Streamlit
Matplotlib


📬 Acknowledgements
Datasets sourced from Kaggle:

FIFA World Cup Dataset by abecklas
FIFA World Rankings by cashncarry
