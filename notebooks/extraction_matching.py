import threading

prompt_step1 = """
From the following given title and abstract, extract all studied exposures and outcomes separately. Ignore any other information.

Extract the following information:

1. "exposures": a list of all studied exposures. Be specific, for example, for salt intake estimated by 24 hour urinary sodium, use 24 hour urinary sodium.Return in full names. And use standard vocabularies.
2. "outcomes": a list of all studied outcomes. Return in full names. And use standard vocabularies.

Return the extracted information in a JSON dictionary.

Example input:
""
Title: "Fatal and nonfatal outcomes, incidence of hypertension, and blood pressure changes in relation to urinary sodium excretion"

Abstract: "Context: Extrapolations from observational studies and short-term intervention trials suggest that population-wide moderation of salt intake might reduce cardiovascular events. Objective: To assess whether 24-hour urinary sodium excretion predicts blood pressure (BP) and health outcomes. Design, setting, and participants: Prospective population study, involving 3681 participants without cardiovascular disease (CVD) who are members of families that were randomly enrolled in the Flemish Study on Genes, Environment, and Health Outcomes (1985-2004) or in the European Project on Genes in Hypertension (1999-2001). Of 3681 participants without CVD, 2096 were normotensive at baseline and 1499 had BP and sodium excretion measured at baseline and last follow-up (2005-2008). Main outcome measures: Incidence of mortality and morbidity and association between changes in BP and sodium excretion. Multivariable-adjusted hazard ratios (HRs) express the risk in tertiles of sodium excretion relative to average risk in the whole study population. Results: Among 3681 participants followed up for a median 7.9 years, CVD deaths decreased across increasing tertiles of 24-hour sodium excretion, from 50 deaths in the low (mean, 107 mmol), 24 in the medium (mean, 168 mmol), and 10 in the high excretion group (mean, 260 mmol; P < .001), resulting in respective death rates of 4.1% (95% confidence interval [CI], 3.5%-4.7%), 1.9% (95% CI, 1.5%-2.3%), and 0.8% (95% CI, 0.5%-1.1%). In multivariable-adjusted analyses, this inverse association retained significance (P = .02): the HR in the low tertile was 1.56 (95% CI, 1.02-2.36; P = .04). Baseline sodium excretion predicted neither total mortality (P = .10) nor fatal combined with nonfatal CVD events (P = .55). Among 2096 participants followed up for 6.5 years, the risk of hypertension did not increase across increasing tertiles (P = .93). Incident hypertension was 187 (27.0%; HR, 1.00; 95% CI, 0.87-1.16) in the low, 190 (26.6%; HR, 1.02; 95% CI, 0.89-1.16) in the medium, and 175 (25.4%; HR, 0.98; 95% CI, 0.86-1.12) in the high sodium excretion group. In 1499 participants followed up for 6.1 years, systolic blood pressure increased by 0.37 mm Hg per year (P < .001), whereas sodium excretion did not change (-0.45 mmol per year, P = .15). However, in multivariable-adjusted analyses, a 100-mmol increase in sodium excretion was associated with 1.71 mm Hg increase in systolic blood pressure (P.<001) but no change in diastolic BP. Conclusions: In this population-based cohort, systolic blood pressure, but not diastolic pressure, changes over time aligned with change in sodium excretion, but this association did not translate into a higher risk of hypertension or CVD complications. Lower sodium excretion was associated with higher CVD mortality.
""

Example output:
{
    "exposures": [“24-hour urinary sodium excretion”],
    "outcomes": [“systolic blood pressure changes”, “diastolic blood pressure changes”, “incidence of hypertension”, “CVD mortality”, “total mortality”, “fatal and nonfatal cardiovascular disease events”]
}

Only output in JSON format, do not include any explanations or additional text in your response. no ```json ``` in your response.
no ```json ``` in your response.
no ```json ``` in your response.
"""

prompt_step2_template = """
From the following given title and abstract, extract the primary relationships between the given exposures and outcomes.

Use the provided entities as references.

Extract the following information:

1. "exposure": the studied exposure
2. "exposure_direction": based on context, classify the direction of the exposure to the level of salt or urinary sodium, whether it's increased or decreased (only increased or decreased) (exposures without explict of decreased expression is an increased). by 'increased' it means higher level of exposure/intervention and 'decreased' it means lower level of exposure/intervention. for example, salt restriction/substitute is a 'decreased'. dietary intervention/education is a decreased (because it changes diet to lower salt intake). but you should also be cautious on for example less dietary intervention/education is an increased (opposite of decreased). refer to instruction in the end of prompt.
3. "outcome": the studied outcome
4. "significance": only return 'positive' if there exists STATISTICALLY significance or 'negative' if no STATISTICAL significance.
5. "direction": return whether th level of outcome increases, decreases, or no_change with the exposure in this relationship (no_change include relationships such as indifferent or unchanged)(only return one of the 3 categories: increase, decrease, no_change)
6. "population_main_condition": disease of studied population
7. "comparator": compared group in relationship
8. "study_design": (only 1 of 7 strict abbreiviation categories: Mendelian randomization ("MR"), Randomized controlled trial("RCT"), Observational study("OS"),Meta analysis ("META"), Review ("REVIEW"), Systematic Review ("SR"),  Others);
9. "included studies": (only return number + RCT/MR/OS)(only for META/REVIEW/SR, strict null for others(RCT,MR,OS));
10. "number of participants": (numeric number only) number of enrolled subjects

Return the extracted information in a list of flat JSON dictionaries of each result.

*Exposure Direction Classification
For each exposure mentioning salt or sodium intake:
		1.	If the intervention or action explicitly decreases sodium/salt (e.g., “salt restriction,” “salt substitute,” “low-sodium diet,” “reduce salt”), set "exposure_direction" to "decreased".
	2.	If it explicitly increases sodium/salt, or uses standard/high salt (e.g., “high-salt diet,” “less salt education,” “usual diet with no reduction”), set "exposure_direction" to "increased".

	Examples of Phrases → exposure_direction:
	•	“Participants in the intervention arm received a salt-substitution product” → decreased
	•	“Control arm participants received the usual advice without further diet restrictions” → increased
	•	“We advised a low-sodium diet” → decreased
	•	“We gave participants fewer educational materials about reducing salt” → increased

Example input entities:
{{
    "exposures": [“24-hour urinary sodium excretion”],
    "outcomes": [“systolic blood pressure changes”, “diastolic blood pressure changes”, “incidence of hypertension”, “CVD mortality”, “total mortality”, “fatal and nonfatal cardiovascular disease events”]
}}

Example input title and abstract:
""
Title: "Fatal and nonfatal outcomes, incidence of hypertension, and blood pressure changes in relation to urinary sodium excretion"

Abstract: "Context: Extrapolations from observational studies and short-term intervention trials suggest that population-wide moderation of salt intake might reduce cardiovascular events. Objective: To assess whether 24-hour urinary sodium excretion predicts blood pressure (BP) and health outcomes. Design, setting, and participants: Prospective population study, involving 3681 participants without cardiovascular disease (CVD) who are members of families that were randomly enrolled in the Flemish Study on Genes, Environment, and Health Outcomes (1985-2004) or in the European Project on Genes in Hypertension (1999-2001). Of 3681 participants without CVD, 2096 were normotensive at baseline and 1499 had BP and sodium excretion measured at baseline and last follow-up (2005-2008). Main outcome measures: Incidence of mortality and morbidity and association between changes in BP and sodium excretion. Multivariable-adjusted hazard ratios (HRs) express the risk in tertiles of sodium excretion relative to average risk in the whole study population. Results: Among 3681 participants followed up for a median 7.9 years, CVD deaths decreased across increasing tertiles of 24-hour sodium excretion, from 50 deaths in the low (mean, 107 mmol), 24 in the medium (mean, 168 mmol), and 10 in the high excretion group (mean, 260 mmol; P < .001), resulting in respective death rates of 4.1% (95% confidence interval [CI], 3.5%-4.7%), 1.9% (95% CI, 1.5%-2.3%), and 0.8% (95% CI, 0.5%-1.1%). In multivariable-adjusted analyses, this inverse association retained significance (P = .02): the HR in the low tertile was 1.56 (95% CI, 1.02-2.36; P = .04). Baseline sodium excretion predicted neither total mortality (P = .10) nor fatal combined with nonfatal CVD events (P = .55). Among 2096 participants followed up for 6.5 years, the risk of hypertension did not increase across increasing tertiles (P = .93). Incident hypertension was 187 (27.0%; HR, 1.00; 95% CI, 0.87-1.16) in the low, 190 (26.6%; HR, 1.02; 95% CI, 0.89-1.16) in the medium, and 175 (25.4%; HR, 0.98; 95% CI, 0.86-1.12) in the high sodium excretion group. In 1499 participants followed up for 6.1 years, systolic blood pressure increased by 0.37 mm Hg per year (P < .001), whereas sodium excretion did not change (-0.45 mmol per year, P = .15). However, in multivariable-adjusted analyses, a 100-mmol increase in sodium excretion was associated with 1.71 mm Hg increase in systolic blood pressure (P.<001) but no change in diastolic BP. Conclusions: In this population-based cohort, systolic blood pressure, but not diastolic pressure, changes over time aligned with change in sodium excretion, but this association did not translate into a higher risk of hypertension or CVD complications. Lower sodium excretion was associated with higher CVD mortality.
""

Example output:
[
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “systolic blood pressure changes”,
        “significance”: “positive”,
        “direction”: “increase”,
        “population_main_condition”: “not found”,
        “comparator”: “baseline sodium excretion”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “diastolic blood pressure changes”,
        “significance”: “negative”,
        “direction”: “no_change”,
        “population_main_condition”: “no_change”,
        “comparator”: “baseline sodium excretion”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “incidence of hypertension”,
        “significance”: “negative”,
        “direction”: “no_change”,
        “population_main_condition”: “not found”,
        “comparator”: “low sodium excretion group”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 2096
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “CVD mortality”,
        “significance”: “positive”,
        “direction”: “decrease”,
        “population_main_condition”: “not found”,
        “comparator”: “low sodium excretion group”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “total mortality”,
        “significance”: “negative”,
        “direction”: “no_change”,
        “population_main_condition”: “not found”,
        “comparator”: “baseline sodium excretion”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }},
    {{
        “exposure”: “24-hour urinary sodium excretion”,
        “exposure_direction”: “increased”,
        “outcome”: “fatal and nonfatal cardiovascular disease events”,
        “significance”: “negative”,
        “direction”: “no_change”,
        “population_main_condition”: “not found”,
        “comparator”: “baseline sodium excretion”,
        “study_design”: “OS”,
        “included_studies”: “null”,
        “number_of_participants”: 3681
    }}
]

Only output in JSON format, do not include any explanations or additional text in your response. no ```json ``` in your response.
no ```json ``` in your response.
no ```json ``` in your response.
"""

# Define token counters
total_input_tokens = 0
total_output_tokens = 0
token_lock = threading.Lock()  # Prevent race conditions in multi-threaded execution

MODEL="deepseek-chat"
client = OpenAI(api_key='sk-...',base_url="https://api.deepseek.com")

# Function to perform the two-step extraction
def two_step_extraction(text):
    global total_input_tokens, total_output_tokens

    # Step 1: Extract exposures and outcomes
    completion_step1 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an evidence-based medicine annotator, particularly in effect of salt or urinary sodium on cardiovascular events, help me extract structured information from free texts of article titles and abstracts"},
            {"role": "user", "content": prompt_step1},
            {"role": "user", "content": text}
        ]
    )

    # Extract token usage
    step1_tokens_in = completion_step1.usage.prompt_tokens
    step1_tokens_out = completion_step1.usage.completion_tokens

    # Update global token counters
    with token_lock:
        total_input_tokens += step1_tokens_in
        total_output_tokens += step1_tokens_out

    # Process step 1 output
    entities_result = completion_step1.choices[0].message.content
    entities_result = entities_result.strip().removeprefix("```json").removesuffix("```").strip()
    entities_dict = json.loads(entities_result)
    entities_json = json.dumps(entities_dict, indent=2)

    # Prepare step 2 prompt
    prompt_step2 = prompt_step2_template.format(entities=entities_json)

    # Step 2: Extract associations using the entities
    completion_step2 = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are an evidence-based medicine annotator, particularly in effect of salt or urinary sodium on cardiovascular events, help me infer associations from given entities and titles and abstracts"},
            {"role": "user", "content": prompt_step2},
            {"role": "user", "content": f"extracted entities: {entities_json}"},
            {"role": "user", "content": f"title and abstract: {text}"}
        ]
    )

    # Extract token usage
    step2_tokens_in = completion_step2.usage.prompt_tokens
    step2_tokens_out = completion_step2.usage.completion_tokens

    # Update global token counters
    with token_lock:
        total_input_tokens += step2_tokens_in
        total_output_tokens += step2_tokens_out

    associations_result = completion_step2.choices[0].message.content
    #time.sleep(3)

    return associations_result

# Worker function to process each row
def worker(input_queue, output_queue, text_column, progress_bar):
    while True:
        index, row = input_queue.get()
        if index is None:
            break
        try:
            output = two_step_extraction(row[text_column])
            extracted_results = literal_eval(output)
            for res in extracted_results:
                res['pmid'] = row['pmid']
                output_queue.put(res)
        except Exception as e:
            print(f"Error processing row {index}: {e}")
            print(f"Input text: {row[text_column][:500]}...")  # Print the first 500 characters of the text
            print(f"Output: {output}")  # Print the output if available
        finally:
            input_queue.task_done()
            progress_bar.update(1)

# Main function to compile extracted results
def compile_extracted_results(input_df, text_column):
    global total_input_tokens, total_output_tokens  # Ensure we modify the global variables

    # Reset token counters at the start of the function
    total_input_tokens = 0
    total_output_tokens = 0

    results = []
    input_queue = Queue()
    output_queue = Queue()

    # Progress bar for processing
    progress_bar = tqdm(total=input_df.shape[0], desc="Processing abstracts", position=0, leave=True)

    # Start worker threads
    for _ in range(20):  # Adjust the number of threads as needed
        threading.Thread(target=worker, args=(input_queue, output_queue, text_column, progress_bar), daemon=True).start()

    # Enqueue rows for processing
    for index, row in input_df.iterrows():
        input_queue.put((index, row))

    # Wait for all tasks to be processed
    input_queue.join()

    # Collect results from output queue
    while not output_queue.empty():
        results.append(output_queue.get())

    # Close progress bar
    progress_bar.close()

    # Print total token usage
    print(f"Total input tokens: {total_input_tokens}")
    print(f"Total output tokens: {total_output_tokens}")
    print(f"Total tokens used (input + output): {total_input_tokens + total_output_tokens}")

    return pd.json_normalize(results)
    return results_df

import openai
import pandas as pd
from tqdm import tqdm
from queue import Queue
import threading
from ast import literal_eval

# Set your OpenAI API key

# Define the target exposure and target outcome for classification
target_exposure = "salt/urinary sodium"
target_outcome = "blood pressure"

system_1 = "You are an expert in epidemiology, specialized in social determinants of health (SDoh), especially in effect of salt on cardiovascular events and mortality. "

prompt_template = f"""I extracted exposures and outcomes from abstracts of studies.
I will send you extracted exposure and extracted outcome in a dictionary.

please help me

1. classify if the extracted exposure concept is 'salt/urinary sodium/sodium chloride/NaCI' or synonyms (must be same or highly associated concept)
2. classify if the extracted outcome concept is 'cardiovascular events/diseases, cvd events/diseases, any kind of mortality or death' or synonyms (must be same or highly associated concept)

note CVD events are major health incidents or outcomes that stem from cardiovascular disease (CVD)—that is, disease of the heart and blood vessels. Common examples include: •  Heart attack (myocardial infarction) •  Stroke (cerebrovascular accident) • Unstable angina •   Sudden cardiac death •  Heart failure exacerbation •    Coronary revascularization procedures (e.g., bypass surgery, angioplasty)

this is the input and output strucuture:
input: {{extracted_exposure: ''; extracted_outcome: ''}}
output: {{exposure_match: 'yes'/'no', outcome_match: 'yes'/'no'}}

below are examples
example 1:
input: {{extracted_exposure: 'high salt intake'; extracted_outcome: 'blood pressure'}}
output: {{exposure_match: 'yes', outcome_match: 'no'}}

example 2:
input: {{extracted_exposure: 'reduced dietary salt'; extracted_outcome: 'diastolic blood pressure'}}
output: {{exposure_match: 'yes', outcome_match: 'no'}}

example 3:
input: {{extracted_exposure: 'exercise'; extracted_outcome: 'myocardial infarction'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

example 4:
input: {{extracted_exposure: 'mobile app intervention'; extracted_outcome: 'stroke'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

example 5:
input: {{extracted_exposure: 'mobile app intervention'; extracted_outcome: 'all cause mortality'}}
output: {{exposure_match: 'no', outcome_match: 'yes'}}

only return the output json, do not output other descriptive words"""

def matching_pair(text, prompt):
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_1},
            {"role": "user", "content": prompt},
            {"role": "user", "content": text}
        ],
        temperature=0,
        response_format={ "type": "json_object" }
    )

    result = completion.choices[0].message.content
    try:
        # Convert the response to a dictionary
        result_dict = literal_eval(result)
        return result_dict
    except Exception as e:
        # Return a dictionary with NA in case of error
        print(result)
        return {"exp_direction": "NA", "exposure_match": "NA", "outcome_match": "NA"}
    #time.sleep(30)

def worker_match(input_queue, output_queue, progress_bar):
    while True:
        index, row = input_queue.get()
        if index is None:
            break
        try:
            result = matching_pair(row['classify_input'], prompt_template)
            # Print input and output for each loop
            # print(f"Input (row {index}): {row['classify_input']}")
            # print(f"Output (row {index}): {result}")

            # Collecting results into output queue with original index
            output_queue.put((index, result))
        except Exception as e:
            print(f"Error processing row {index}: {e}")
        finally:
            input_queue.task_done()
            progress_bar.update(1)

# Main function to process all rows in the DataFrame using multithreading
def process_salt_results(df, prompt_template):
    input_queue = Queue()
    output_queue = Queue()

    # Progress bar for processing
    progress_bar = tqdm(total=df.shape[0], desc="Processing rows", position=0, leave=True)

    # Start worker threads
    num_workers = 16
    for _ in range(num_workers):
        threading.Thread(target=worker_match, args=(input_queue, output_queue, progress_bar), daemon=True).start()

    # Enqueue rows for processing
    for index, row in df.iterrows():
        input_queue.put((index, row))

    # Wait for all tasks to be processed
    input_queue.join()

    # Collect results from output queue
    results = []
    while not output_queue.empty():
        index, result = output_queue.get()
        # Ensure each result retains the correct index
        #df.at[index, 'exp_direction'] = result['exp_direction']
        df.at[index, 'exposure_match'] = result['exposure_match']
        df.at[index, 'outcome_match'] = result['outcome_match']

    # Close progress bar
    progress_bar.close()

    return df

all_got_df_final_step_1 = compile_extracted_results(all_got_df, 'input')##
all_got_df_final_step_2 = process_salt_results(all_got_df_final_step_1, prompt_template)
