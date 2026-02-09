### Steps to run

1. Run the following command from the project root directory:

```bash
python prepare_data_calendar.py --model="meta-llama/llama-3.1-8b-instruct" --data_path="calendar_scheduling_input.json" --out_path="calendar_scheduling_output.json"
```
2. Once "calendar_scheduling_output.json" is created, run below command for evaluation:

```bash
python evaluate_calendar_scheduling.py --data_path="calendar_scheduling_output.json
```
