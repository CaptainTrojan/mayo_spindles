import optuna
import os

POSTGRES_PW = os.getenv("POSTGRES_PW")
if POSTGRES_PW is None:
    raise ValueError("Please set the POSTGRES_PW environment variable to the password of the PostgreSQL database.")


study_summaries = optuna.get_all_study_summaries(
    storage=f"postgresql://postgres:{POSTGRES_PW}@147.228.127.28:40442",
    include_best_trial=True,
)

for summary in study_summaries:
    # n_trials, user_attrs, start_datetime
    print(f"Study name: {summary.study_name}")
    print(f"Number of trials: {summary.n_trials}")
    print(f"User attributes: {summary.user_attrs}")
    print(f"Start datetime: {summary.datetime_start}")
    print(f"Best trial: {summary.best_trial}")
    print("=" * 80)