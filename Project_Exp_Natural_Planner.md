# NATURAL PLAN - Detailed Project Explanation

## 📋 Table of Contents
1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Data Flow](#data-flow)
4. [File-by-File Explanation](#file-by-file-explanation)
5. [Reading Sequence](#reading-sequence)

---

## 🎯 Project Overview

**Natural Plan** is a **benchmarking system** for evaluating Large Language Models (LLMs) on three types of planning tasks:

1. **Trip Planning** - Planning multi-city European trips
2. **Meeting Planning** - Scheduling meetings with multiple people across different locations
3. **Calendar Scheduling** - Finding common meeting times for multiple participants

### Purpose
- Test how well LLMs can solve real-world planning problems
- Compare model performance across different complexity levels
- Provide standardized evaluation metrics

---

## 🏗️ Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    NATURAL PLAN SYSTEM                       │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌─────────────┐│
│  │   DATA       │      │  EVALUATION  │      │   RESULTS   ││
│  │   LAYER      │─────▶│    LAYER     │─────▶│   LAYER     ││
│  └──────────────┘      └──────────────┘      └─────────────┘│
│         │                      │                      │      │
│         │                      │                      │      │
│  ┌──────▼──────┐      ┌────────▼────────┐    ┌───────▼────┐│
│  │ JSON Files  │      │  Parser +       │    │  Accuracy  ││
│  │ - Trip      │      │  Validator      │    │  Metrics   ││
│  │ - Meeting   │      │  Functions      │    │  Reports   ││
│  │ - Calendar  │      │                 │    │            ││
│  └─────────────┘      └─────────────────┘    └────────────┘│
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

### Component Breakdown

#### 1. **Data Layer** (`data/` directory)
- **Purpose**: Stores test cases with prompts, ground truth solutions, and model responses
- **Format**: JSON files
- **Contents**:
  - `prompt_0shot`: Zero-shot prompts (no examples)
  - `prompt_5shot`: Five-shot prompts (with examples)
  - `golden_plan`: Correct/optimal solution
  - `pred_5shot_pro`: Model's response (from Gemini 1.5 Pro)

#### 2. **Evaluation Layer** (Python scripts)
- **Purpose**: Parse model responses and compare with ground truth
- **Components**:
  - **Parsers**: Extract structured information from natural language
  - **Validators**: Check if plans satisfy constraints
  - **Scorers**: Calculate accuracy metrics

#### 3. **Results Layer** (Console output)
- **Purpose**: Display evaluation metrics
- **Metrics**: Accuracy, solve rate, breakdown by complexity

---

## 🔄 Data Flow

### Complete Evaluation Flow

```
1. LOAD DATA
   │
   ├─▶ Read JSON file (e.g., trip_planning.json)
   │
   └─▶ Extract for each example:
       - Model response (pred_5shot_pro)
       - Ground truth (golden_plan)
       - Constraints/metadata
   
2. PARSE RESPONSES
   │
   ├─▶ Trip Planning:
   │   └─▶ Extract: cities, flight days, stay durations
   │
   ├─▶ Meeting Planning:
   │   └─▶ Extract: travel steps, meeting steps, times
   │
   └─▶ Calendar Scheduling:
       └─▶ Extract: day, start time, end time
   
3. VALIDATE PLANS
   │
   ├─▶ Check constraints:
   │   - Time windows
   │   - Locations
   │   - Durations
   │   - Logical consistency
   │
   └─▶ Score each plan (0 or 1)
   
4. COMPUTE METRICS
   │
   ├─▶ Count correct plans
   │
   ├─▶ Calculate accuracy = correct / total
   │
   └─▶ Break down by complexity
   
5. OUTPUT RESULTS
   │
   └─▶ Print accuracy metrics
```

### Detailed Flow for Each Task

#### **Trip Planning Flow**
```
JSON Data
  │
  ├─▶ Extract: cities, durations, pred_5shot_pro
  │
  └─▶ parse_response()
      │
      ├─▶ Find flight patterns: "Day X from CityA to CityB"
      ├─▶ Find visit patterns: "1-5" (days 1 to 5)
      └─▶ Calculate: (city, stay_days) tuples
      │
      └─▶ compute_example_score()
          │
          ├─▶ Compare cities: exact match?
          ├─▶ Compare durations: exact match?
          └─▶ Return: 1.0 (match) or 0.0 (mismatch)
```

#### **Meeting Planning Flow**
```
JSON Data
  │
  ├─▶ Extract: constraints, dist_matrix, pred_5shot_pro
  │
  └─▶ parse_text_plan()
      │
      └─▶ Split by "." → list of steps
      │
      └─▶ validator_from_text()
          │
          ├─▶ Track: current_location, current_time
          ├─▶ For each step:
          │   ├─▶ "You travel" → update location & time
          │   ├─▶ "You wait" → update time
          │   └─▶ "You meet X" → validate:
          │       ├─▶ Correct location?
          │       ├─▶ Within time window?
          │       ├─▶ Correct duration?
          │       └─▶ Not duplicate meeting?
          │
          └─▶ Return: number of valid meetings
          │
          └─▶ Compare with golden_plan score
              └─▶ Match? → Accuracy = 1, else 0
```

#### **Calendar Scheduling Flow**
```
JSON Data
  │
  ├─▶ Extract: num_people, num_days, pred_5shot_pro, golden_plan
  │
  └─▶ _parse_response()
      │
      ├─▶ Regex: "Monday, 9:00 - 10:30"
      ├─▶ Extract: day, start_hour, end_hour
      └─▶ Convert hours to numbers (9:00 → 9.0, 9:30 → 9.5)
      │
      └─▶ compute_solve_rate()
          │
          ├─▶ Parse model response
          ├─▶ Parse golden solution
          └─▶ Compare: exact match on all three?
              └─▶ Return: 1.0 (match) or 0.0 (mismatch)
```

---

## 📁 File-by-File Explanation

### **1. README.md** (Root)
**Purpose**: Project overview and quick start guide

**Key Sections**:
- Project description
- Installation instructions
- Usage examples
- Citation information

**What to learn**: Overall project purpose and how to get started

---

### **2. data/README.md**
**Purpose**: Explains the data structure

**Key Information**:
- Field descriptions for each task
- Data format specifications
- Usage instructions

**What to learn**: What data is stored and how it's structured

---

### **3. evaluate_trip_planning.py**
**Purpose**: Evaluates trip planning responses

**Key Functions**:

#### `parse_response(response: str)`
- **Input**: Raw model response text
- **Process**: 
  - Uses regex to find flight patterns: `r'.*Day (\d+).*from (\w+) to (\w+)'`
  - Finds visit day ranges: `r'\d+-\d+'`
  - Calculates stay duration per city
- **Output**: List of `(city, stay_days)` tuples
- **Example**: `[("Paris", 3), ("London", 2), ("Rome", 4)]`

#### `compute_example_score(cities, durations, parsed_plan)`
- **Input**: 
  - `cities`: "Paris**London**Rome"
  - `durations`: "3**2**4"
  - `parsed_plan`: `[("Paris", 3), ("London", 2), ("Rome", 4)]`
- **Process**: 
  - Splits cities and durations by `**`
  - Compares element by element
  - Must match exactly (city name AND duration)
- **Output**: `1.0` (exact match) or `0.0` (mismatch)

#### `compute_score(cities, durations, responses)`
- **Input**: Lists of all examples
- **Process**: 
  - Parses all responses
  - Scores each example
  - Calculates average accuracy
- **Output**: Overall accuracy (0.0 to 1.0)

#### `main()`
- **Flow**:
  1. Load JSON data
  2. Extract cities, durations, responses
  3. Call `compute_score()`
  4. Print accuracy

---

### **4. evaluate_meeting_planning.py**
**Purpose**: Evaluates meeting planning responses

**Key Functions**:

#### `convert_to_time_obj(time_str: str)`
- **Input**: "9:00AM"
- **Output**: `datetime.datetime` object
- **Purpose**: Convert string times to comparable objects

#### `process_constraints(data)`
- **Input**: Raw constraint tuples
- **Process**: 
  - Extracts: name, location, time window, meeting duration
  - Converts times to datetime objects
- **Output**: Dictionary with processed constraints
- **Example**:
  ```python
  {
    "Stephanie": {
      "location": "Mission District",
      "start_time": datetime(9:00AM),
      "end_time": datetime(1:30PM),
      "meeting_time": 120  # minutes
    }
  }
  ```

#### `parse_text_plan(plan: str)`
- **Input**: Raw plan text
- **Process**: 
  - Removes "SOLUTION:" prefix if present
  - Splits by "." to get individual steps
  - Cleans whitespace
- **Output**: List of step strings
- **Example**: 
  ```python
  [
    "You start at Marina District at 9:00AM",
    "You travel to Mission District in 20 minutes",
    "You meet Stephanie for 120 minutes from 10:30AM to 12:30PM"
  ]
  ```

#### `validator_from_text(plan, constraints, start_location, initial_time, dist_matrix)`
- **Purpose**: Validates a plan step-by-step
- **State Tracking**:
  - `cur_location`: Current location
  - `cur_time`: Current time
  - `met_with`: Set of people already met
  - `score`: Number of valid meetings
- **Step Processing**:
  - **"You start"**: Initialize state
  - **"You travel to X in Y minutes"**: 
    - Update location
    - Add travel time to current time
  - **"You wait until TIME"**: 
    - Check time doesn't go backwards
    - Update current time
  - **"You meet X for Y minutes"**: 
    - Check person not already met
    - Check correct location
    - Check within time window
    - Check meeting fits in available time
    - If valid: increment score, update time
- **Output**: Number of valid meetings scheduled

#### `main()`
- **Flow**:
  1. Load JSON data
  2. For each example:
     - Parse model response
     - Validate model plan → get score
     - Validate golden plan → get golden score
     - Compare: if scores match → accuracy = 1
  3. Aggregate by number of people
  4. Print accuracy breakdown

---

### **5. evaluate_calendar_scheduling.py**
**Purpose**: Evaluates calendar scheduling responses

**Key Functions**:

#### `hour_to_num(hr_str)`
- **Input**: "9:00" or "9:30"
- **Process**: 
  - Extracts hour
  - Adds 0.5 if minutes = 30
- **Output**: `9.0` or `9.5`
- **Purpose**: Convert time strings to comparable numbers

#### `_parse_response(response: str)`
- **Input**: Raw model response
- **Process**: 
  - Regex: `r'[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+'`
  - Matches: "Monday, 9:00 - 10:30"
  - Extracts: day, start_hour, end_hour
  - Converts hours to numbers
- **Output**: `(day, start_hour_num, end_hour_num)`
- **Example**: `("Monday", 9.0, 10.5)`

#### `compute_solve_rate(responses, solutions)`
- **Input**: Lists of model responses and golden solutions
- **Process**: 
  - Parse each response
  - Parse each solution
  - Compare: day, start_hour, end_hour must all match exactly
- **Output**: Fraction of exact matches (0.0 to 1.0)

#### `main()`
- **Flow**:
  1. Load JSON data
  2. Extract responses and solutions
  3. Calculate overall solve rate
  4. Calculate solve rate by complexity (people × days)
  5. Print results

---

## 📖 Reading Sequence

### **Phase 1: Understanding the Project (Start Here)**

#### Step 1: `README.md` (Root)
- **Why**: Get overall project context
- **Focus on**: 
  - What the project does
  - What the three tasks are
  - Installation and usage

#### Step 2: `data/README.md`
- **Why**: Understand the data structure
- **Focus on**: 
  - What fields exist in each task
  - What `prompt_5shot` and `golden_plan` mean
  - How data is organized

---

### **Phase 2: Understanding the Simplest Task**

#### Step 3: `evaluate_calendar_scheduling.py`
- **Why**: Simplest evaluation logic (just parsing and comparison)
- **Read in this order**:
  1. `main()` function (lines 88-120) - See the overall flow
  2. `hour_to_num()` (lines 31-34) - Simple utility
  3. `_parse_response()` (lines 37-61) - How it extracts time info
  4. `compute_solve_rate()` (lines 64-85) - How it compares

**Key Learning**: 
- How regex is used to extract structured data
- How exact matching works
- Simple evaluation pattern

---

### **Phase 3: Understanding Medium Complexity Task**

#### Step 4: `evaluate_trip_planning.py`
- **Why**: More complex parsing with multiple patterns
- **Read in this order**:
  1. `main()` function (lines 137-150) - Overall flow
  2. `parse_response()` (lines 32-82) - Complex regex parsing
  3. `compute_example_score()` (lines 85-111) - Scoring logic
  4. `compute_score()` (lines 114-134) - Aggregation

**Key Learning**:
- Multiple regex patterns
- Building structured data from text
- Element-by-element comparison

---

### **Phase 4: Understanding Most Complex Task**

#### Step 5: `evaluate_meeting_planning.py`
- **Why**: Most complex - stateful validation with constraints
- **Read in this order**:
  1. `main()` function (lines 243-278) - Overall flow
  2. `convert_to_time_obj()` (lines 37-38) - Time conversion
  3. `process_constraints()` (lines 41-51) - Constraint processing
  4. `parse_text_plan()` (lines 228-240) - Text parsing
  5. `validator_from_text()` (lines 54-128) - **MOST IMPORTANT**
     - Read carefully: This is the core validation logic
     - Understand state tracking
     - Understand each step type

**Key Learning**:
- Stateful validation
- Constraint checking
- Time and location tracking
- Error handling

---

### **Phase 5: Understanding Data Structure**

#### Step 6: Examine Sample Data
- **Why**: See actual data format
- **How**: 
  ```bash
  # Look at a sample entry
  python3 -c "import json; data=json.load(open('data/trip_planning.json')); print(json.dumps(list(data.values())[0], indent=2))"
  ```

**Key Learning**:
- Actual JSON structure
- What prompts look like
- What model responses look like
- What golden plans look like

---

## 🎓 Key Concepts to Understand

### 1. **Exact Match Evaluation**
- All three tasks use exact matching
- Model response must match golden solution exactly
- No partial credit

### 2. **Parsing Natural Language**
- Models output natural language
- Evaluation scripts parse this into structured data
- Uses regex patterns and string manipulation

### 3. **Constraint Validation**
- Meeting Planning validates:
  - Time windows
  - Locations
  - Travel times
  - Logical consistency

### 4. **State Tracking**
- Meeting Planning tracks:
  - Current location
  - Current time
  - People already met
- Updates state as plan progresses

### 5. **Error Handling**
- Invalid plans cause errors
- Errors are printed but don't crash
- Invalid plans get score of 0

---

## 🔍 How to Trace Through Code

### Example: Trip Planning Evaluation

1. **Start**: `python3 evaluate_trip_planning.py`
2. **Entry Point**: `main()` function
3. **Data Loading**: `json.load()` reads the JSON file
4. **For each example**:
   - Extract `pred_5shot_pro` (model response)
   - Extract `cities` and `durations` (ground truth)
5. **Parsing**: `parse_response()` converts text to structure
6. **Scoring**: `compute_example_score()` compares parsed plan to ground truth
7. **Aggregation**: `compute_score()` averages all scores
8. **Output**: Print accuracy

### Example: Meeting Planning Validation

1. **Start**: `python3 evaluate_meeting_planning.py`
2. **Entry Point**: `main()` function
3. **For each example**:
   - Parse model response → list of steps
   - Initialize validator state
   - Process each step:
     - "You travel" → update location & time
     - "You meet X" → validate constraints
   - Count valid meetings
   - Compare count with golden plan count
4. **Output**: Print accuracy by number of people

---

## 💡 Tips for Understanding

1. **Start Simple**: Begin with Calendar Scheduling (simplest)
2. **Read Functions Top-to-Bottom**: Follow the execution flow
3. **Print Debugging**: Add `print()` statements to see intermediate values
4. **Test with One Example**: Modify code to process just one example
5. **Read Comments**: Code has good comments explaining logic
6. **Understand Regex**: Learn basic regex patterns used
7. **Trace State**: For Meeting Planning, trace how state changes

---

## 🚀 Next Steps

After understanding the code:

1. **Modify Evaluation**: Try different scoring methods
2. **Add New Tasks**: Create evaluation for new planning tasks
3. **Improve Parsing**: Handle edge cases in model responses
4. **Visualize Results**: Create charts/graphs of accuracy
5. **Test with Your Model**: Replace `pred_5shot_pro` with your model's responses

---





