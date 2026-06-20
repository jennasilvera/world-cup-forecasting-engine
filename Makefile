PYTHON = python
RESULTS = data/sample/historical_results_sample.csv
HOME_TEAM = Argentina
AWAY_TEAM = France
N_SIMULATIONS = 500
HOME_ODDS = 2.20
DRAW_ODDS = 3.40
AWAY_ODDS = 3.50
FINAL_HOME_SCORE = 1
FINAL_AWAY_SCORE = 1
CLOSING_HOME_ODDS = 2.10
CLOSING_DRAW_ODDS = 3.25
CLOSING_AWAY_ODDS = 3.60

.PHONY: install lint test check health ingest build-elo build-features backtest report-backtest poisson report-match simulate-group-stage report-group-stage demo clean

install:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e .

lint:
	ruff check .

test:
	pytest

check: lint test

health:
	$(PYTHON) -m wc_forecast health

ingest:
	$(PYTHON) -m wc_forecast ingest-results $(RESULTS)

build-elo:
	$(PYTHON) -m wc_forecast build-elo

build-features:
	$(PYTHON) -m wc_forecast build-features

backtest:
	$(PYTHON) -m wc_forecast backtest-logistic

report-backtest:
	$(PYTHON) -m wc_forecast report-backtest

poisson:
	$(PYTHON) -m wc_forecast predict-poisson $(HOME_TEAM) $(AWAY_TEAM)

report-match:
	$(PYTHON) -m wc_forecast report-match $(HOME_TEAM) $(AWAY_TEAM)

market-edge:
	$(PYTHON) -m wc_forecast evaluate-market $(HOME_TEAM) $(AWAY_TEAM) --home-odds $(HOME_ODDS) --draw-odds $(DRAW_ODDS) --away-odds $(AWAY_ODDS)

log-prediction:
	$(PYTHON) -m wc_forecast log-prediction $(HOME_TEAM) $(AWAY_TEAM) --home-odds $(HOME_ODDS) --draw-odds $(DRAW_ODDS) --away-odds $(AWAY_ODDS)

settle-prediction:
	@PREDICTION_ID=$$(tail -n 1 outputs/prediction_ledger.csv | cut -d',' -f1); \
	$(PYTHON) -m wc_forecast settle-prediction "$$PREDICTION_ID" \
	  --final-home-score $(FINAL_HOME_SCORE) \
	  --final-away-score $(FINAL_AWAY_SCORE) \
	  --closing-home-odds $(CLOSING_HOME_ODDS) \
	  --closing-draw-odds $(CLOSING_DRAW_ODDS) \
	  --closing-away-odds $(CLOSING_AWAY_ODDS)

simulate-group-stage:
	$(PYTHON) -m wc_forecast simulate-group-stage --n-simulations $(N_SIMULATIONS)

report-group-stage:
	$(PYTHON) -m wc_forecast report-group-stage

demo: ingest build-elo build-features backtest report-backtest poisson report-match market-edge log-prediction settle-prediction simulate-group-stage report-group-stage
	@echo "Demo pipeline complete."

clean:
	rm -f outputs/*.csv
	rm -f reports/logistic_backtest_report.md
	rm -f reports/match_prediction_report.md
	rm -f reports/group_stage_simulation_report.md
	rm -f data/processed/*.csv
