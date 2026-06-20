PYTHON = python
RESULTS = data/sample/historical_results_sample.csv
HOME_TEAM = Argentina
AWAY_TEAM = France
N_SIMULATIONS = 500

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

simulate-group-stage:
	$(PYTHON) -m wc_forecast simulate-group-stage --n-simulations $(N_SIMULATIONS)

report-group-stage:
	$(PYTHON) -m wc_forecast report-group-stage

demo: ingest build-elo build-features backtest report-backtest poisson report-match simulate-group-stage report-group-stage
	@echo "Demo pipeline complete."

clean:
	rm -f outputs/*.csv
	rm -f reports/logistic_backtest_report.md
	rm -f reports/match_prediction_report.md
	rm -f reports/group_stage_simulation_report.md
	rm -f data/processed/*.csv
