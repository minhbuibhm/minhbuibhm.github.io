#!/usr/bin/env python
# coding: utf-8

# # L5: Generate Data & Finetune

# > Note: You can access the `data` and `util` subdirectories used in the course. In Jupyter version 6, this is via the File>Open menu. In Jupyter version 7 this is in View> File Browser
# 
# > Also note that as models and systems change, the output of the models may shift from the video content.

# In[1]:


from dotenv import load_dotenv
_ = load_dotenv()   #load environmental variable LAMINI_API_KEY with key from .env file


# In[4]:


import os
print(os.getenv("LAMINI_API_KEY")) # eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJhcHAiLCJleHAiOjE3OTk5OTk5OTksInN1YiI6MjE1NzIyMCwiYXVkIjoiV0VCIiwiaWF0IjoxNjk0MDc2ODUxfQ.YWvlcdnvCaNFKsM-r2hAkKCpXHLIB1Yw3aZDUn1Mt4s


# In[ ]:


import lamini


# In[ ]:


import logging
import random
from typing import AsyncIterator, Iterator, Union
import sqlite3
import copy
from tqdm import tqdm

import pandas as pd
import jsonlines
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline
from util.get_schema import get_schema, get_schema_s
from util.make_llama_3_prompt import make_llama_3_prompt
from util.setup_logging import setup_logging

logger = logging.getLogger(__name__)
engine = sqlite3.connect("./nba_roster.db")
setup_logging()

class Args:
    def __init__(self, 
                 max_examples=100, 
                 sql_model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
                 gold_file_name="gold-test-set.jsonl",
                 training_file_name="generated_queries.jsonl",
                 num_to_generate=10):
        self.sql_model_name = sql_model_name
        self.max_examples = max_examples
        self.gold_file_name = gold_file_name
        self.training_file_name = training_file_name
        self.num_to_generate = num_to_generate


# ## Working Backwards from what you have:
# ### <font color="blue">First</font>: From Scheme and example, generate <font color="blue">new SQL queries</font> 

# In[ ]:


system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
system += (
    "Consider a table called 'nba_roster' with the following schema (columns)\n"
)
system += get_schema_s()
system += "Consider the following questions, and queries used to answer them:\n"


# In[ ]:


system


# In[ ]:


question = """What is the median weight in the NBA?"""
sql = "select CAST(SUBSTR(WT, 1, INSTR(WT,' ')) as INTEGER) as percentile from nba_roster order by percentile limit 1 offset (select count(*) from nba_roster)/2;"

system += "Question: " + question + "\n"
system += "Query: " + sql + "\n"


# In[ ]:


print(system)


# In[ ]:


user = "Write two queries that are similar but different to those above.\n"
user += "Format the queries as a JSON object, i.e.\n"
user += '{ "explanation": str, "sql_query_1" : str, "sql_query_2": str }.\n'


# In[ ]:


print(user)


# In[ ]:


user += "First write an explanation of why you decided to write these new queries in about 3-5 sentences, then write valid sqlite SQL queries for each of the 2 new queries. Make sure each query is complete and ends with a ;\n"


# In[ ]:


print(user)


# In[ ]:


prompt = make_llama_3_prompt(user, system)


# In[ ]:


llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")
result = llm.generate(prompt, output_type={ "explanation": "str", "sql_query_1" : "str", "sql_query_2": "str" }, max_new_tokens=200)
print(result)


# In[ ]:


def check_sql_query(query):
    try:
        pd.read_sql(query, con=engine)
    except Exception as e:
        logger.debug(f"Error in SQL query: {e}")
        return False

    logger.info(f"SQL query {query} is valid")

    return True


# In[ ]:


check_sql_query(result["sql_query_1"])


# In[ ]:


check_sql_query(result["sql_query_2"])


# In[ ]:


# Wrap it all up together in a class

class ModelStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=300,
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        prompt = self.add_template(prompt)

        results = super().generate(
            prompt,
            output_type={
                "explanation": "str",
                "sql_query_1": "str",
                "sql_query_2": "str",
            },
            *args,
            **kwargs,
        )

        return results

    async def add_template(self, prompts):
        async for prompt in prompts:
            new_prompt = make_llama_3_prompt(**self.make_prompt(prompt.data))
            yield PromptObject(prompt=new_prompt, data=prompt.data)

    async def process_results(self, results):
        async for result in results:
            if result is None:
                continue

            if result.response is None:
                continue

            logger.info("=====================================")
            logger.info(f"Generated query 1: {result.response['sql_query_1']}")
            logger.info(f"Generated query 2: {result.response['sql_query_2']}")
            logger.info("=====================================")

            if self.check_sql_query(result.response["sql_query_1"]):
                new_result = PromptObject(prompt="", data=copy.deepcopy(result.data))
                new_result.data.generated_sql_query = result.response["sql_query_1"]
                yield new_result

            if self.check_sql_query(result.response["sql_query_2"]):
                new_result = PromptObject(prompt="", data=copy.deepcopy(result.data))
                new_result.data.generated_sql_query = result.response["sql_query_2"]
                yield new_result

    def make_prompt(self, data):
        system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
        system += (
            "Consider a table called 'nba_roster' with the following schema (columns)\n"
        )
        system += get_schema()
        system += "Consider the following questions, and queries used to answer them:\n"
        for example in data.sample:
            system += "Question: " + example["question"] + "\n"
            system += "Query: " + example["sql"] + "\n"

        # Important: generate relevant queries to your reference data
        # Ideally, close to those that are failing so you can show the model examples of how to do it right!
        user = "Write two queries that are similar but different to those above.\n"
        user += "Format the queries as a JSON object, i.e.\n"
        user += '{ "explanation": str, "sql_query_1" : str, "sql_query_2": str }.\n'

        # Next, use Chain of Thought (CoT) and prompt-engineering to help with generating SQL queries
        user += "First write an explanation of why you decided to write these new queries in about 3-5 sentences, then write valid sqlite SQL queries for each of the 2 new queries. Make sure each query is complete and ends with a ;\n"

        return {"system": system, "user": user}

    def check_sql_query(self, query):
        try:
            pd.read_sql(query, con=engine)
        except Exception as e:
            logger.debug(f"Error in SQL query: {e}")
            return False

        logger.info(f"SQL query {query} is valid")

        return True


# ### <font color="blue">Second:</font> Now that you have queries, <font color="blue">generate questions</font> for those queries
# 

# In[ ]:


system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
system += (
    "Consider a table called 'nba_roster' with the following schema (columns)\n"
)
system += get_schema() + "\n"
system += "Queries, and questions that they are used to answer:\n"

example_question = """What is the median weight in the NBA?"""
example_sql = "select CAST(SUBSTR(WT, 1, INSTR(WT,' ')) as INTEGER) as percentile from nba_roster order by percentile limit 1 offset (select count(*) from nba_roster)/2;"

system += "Question: " + example_question + "\n"
system += "Query: " + example_sql + "\n"


# In[ ]:


generated_sql = result["sql_query_2"]


# In[ ]:


user = "Now consider the following query.\n"
user += "Query: " + generated_sql + "\n"
user += "Write a question that this query could be used to answer.\n"


# In[ ]:


user += "Format your response as a JSON object, i.e.\n"
user += '{ "explanation": str, "question": str }.\n'

user += "First write an explanation in about 3-5 sentences, then write a one sentence question.\n"


# In[ ]:


prompt = make_llama_3_prompt(user, system)
result = llm.generate(prompt, output_type={ "explanation": "str", "question" : "str" }, max_new_tokens=200)
print(result)


# In[ ]:


# Wrap it all up together in a class which generates a question
# given a query

class QuestionStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super().generate(
            prompt,
            output_type={
                "explanation": "str",
                "question": "str",
            },
            *args,
            **kwargs,
        )
        return results

    def preprocess(self, obj: PromptObject):
        new_prompt = make_llama_3_prompt(**self.make_question_prompt(obj.data))
        obj.prompt = new_prompt

    def make_question_prompt(self, data):
        system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
        system += (
            "Consider a table called 'nba_roster' with the following schema (columns)\n"
        )
        system += get_schema() + "\n"
        system += "Queries, and questions that they are used to answer:\n"
        for example in data.sample:
            system += "Query: " + example["sql"] + "\n"
            system += "Question: " + example["question"] + "\n"

        user = "Now consider the following query.\n"
        user += "Query: " + data.generated_sql_query + "\n"
        user += "Write a question that this query could be used to answer.\n"

        # Using Chain of Thought (CoT) again
        # This time you can do it programmatically with function calling, so you can easily extract a question out of the JSON object
        user += "Format your response as a JSON object, i.e.\n"
        user += '{ "explanation": str, "question": str }.\n'

        user += "First write an explanation in about 3-5 sentences, then write a one sentence question.\n"

        return {"system": system, "user": user}


# In[ ]:


class QueryGenPipeline(GenerationPipeline):
    def __init__(self):
        super().__init__()
        self.model_stage = ModelStage()
        self.question_stage = QuestionStage()

    def forward(self, x):
        x = self.model_stage(x)
        x = self.question_stage(x)
        return x


# In[ ]:


async def run_query_gen_pipeline(gold_queries):
    return QueryGenPipeline().call(gold_queries)


# In[ ]:


# Generate N samples, for every example in the gold dataset

all_examples = []

async def load_gold_queries(args):
    path = f"data/{args.gold_file_name}"

    with jsonlines.open(path) as reader:
        global all_examples

        all_examples = [obj for obj in reader]

    sample_count = args.num_to_generate
    sample_size = 3

    random.seed(42)

    for i in range(sample_count):
        example_sample = ExampleSample(random.sample(all_examples, sample_size), i)
        yield PromptObject(prompt="", data=example_sample)


class ExampleSample:
    def __init__(self, sample, index):
        self.sample = sample
        self.index = index


# In[ ]:


async def save_generation_results(results, args):
    path = f"data/training_data/{args.training_file_name}"

    pbar = tqdm(desc="Saving results", unit=" results")
    with jsonlines.open(path, "w") as writer:

        async for result in results:
            writer.write(
                {
                    "question": result.response["question"],
                    "sql": result.data.generated_sql_query,
                }
            )
            pbar.update()

        for example in all_examples:
            writer.write(example)
            pbar.update()


# In[ ]:


args = Args()
gold_queries = load_gold_queries(args)
results = await run_query_gen_pipeline(gold_queries)
await save_generation_results(results, args)


# display the queries just generated above

# In[ ]:


#!cat "data/training_data/generated_queries.jsonl"


# display the archived queries which match the course video.

# In[ ]:


get_ipython().system('cat "data/training_data/archive/generated_queries.jsonl"')


# ### Round of finetuning
# Now that you have data, even if it is not perfect, go through a round of finetuning!

# In[ ]:


import logging
import os
from datetime import datetime
from pprint import pprint
from typing import AsyncIterator, Iterator, Union
import sqlite3
from tqdm import tqdm

import pandas as pd
import jsonlines
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline
from util.get_schema import get_schema
from util.make_llama_3_prompt import make_llama_3_prompt
from util.setup_logging import setup_logging
from util.load_dataset import get_dataset
from util.get_default_finetune_args import get_default_finetune_args

logger = logging.getLogger(__name__)
engine = sqlite3.connect("./nba_roster.db")
setup_logging()

class Args:
    def __init__(self, 
                 max_examples=100, 
                 sql_model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
                 gold_file_name="gold-test-set.jsonl",
                 training_file_name="archive/generated_queries.jsonl",
                 num_to_generate=10):
        self.sql_model_name = sql_model_name
        self.max_examples = max_examples
        self.gold_file_name = gold_file_name
        self.training_file_name = training_file_name
        self.num_to_generate = num_to_generate


# make_question will take the questions and queries from the training_file and embed them in the prompt below to form the training data.

# In[ ]:


def make_question(obj):
    system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
    system += "Consider the nba_roster table with the following schema:\n"
    system += get_schema() + "\n"
    system += (
        "Write a sqlite SQL query that would help you answer the following question:\n"
    )
    user = obj["question"]
    return {"system": system, "user": user}


# In[ ]:


args = Args()
llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")


# In[ ]:


dataset = get_dataset(args, make_question)


# In[ ]:


finetune_args = get_default_finetune_args()


# This fine tuning step takes about 30 mintues to complete. The dispatch to run on the lamini services is commented out and the pre-computed final results of the run are provided below. You can uncomment and run if you have modified data on your own.

# In[ ]:


#llm.train(
#    data_or_dataset_id=dataset,
#    finetune_args=finetune_args,
#    is_public=True,  # For sharing
#)


# We can examine this pre-computed finetuning result.

# In[ ]:


llm = lamini.Lamini(model_name="a5ebf1c4879569101f32444afae5adcafbfce9c5a6ed13035fd892147f7d59bc")


# In[ ]:


question = """Who is the highest paid NBA player?"""
system = f"""You are an NBA analyst with 15 years of experience writing complex SQL queries. Consider the nba_roster table with the following schema:
{get_schema()}

Write a sqlite query to answer the following question. Follow instructions exactly"""
prompt = make_llama_3_prompt(question, system)
print("Question:\n", question)


# In[ ]:


print("Answer:")
print(llm.generate(prompt, max_new_tokens=200))


# In[ ]:


query="SELECT salary, name FROM nba_roster WHERE salary != '--' ORDER BY CAST(REPLACE(REPLACE(salary, '$', ''), ',','') AS INTEGER) DESC LIMIT 1;"
df = pd.read_sql(query, con=engine)
print(df)


# Now lets run an evaluation over the eval dataset. Load code from lesson 3.

# In[ ]:


# Collapsible or utils from Lesson 3 Lab for evaluation
class QueryStage(GenerationNode):
    def __init__(self, model_name):
        super().__init__(
            model_name=model_name,
            max_new_tokens=300,
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super().generate(
            prompt,
            output_type={"sqlite_query": "str"},
            *args,
            **kwargs,
        )
        return results


    def postprocess(self, obj: PromptObject):
        # Run both the generated and reference (Gold Dataset) SQL queries
        # Assessing whether the SQL queries succeeded in hitting the database (not correctness yet!)
        
        query_succeeded = False

        try:
            logger.info(f"Running SQL query '{obj.response['sqlite_query']}'")
            obj.data["generated_query"] = obj.response["sqlite_query"]
            df = pd.read_sql(obj.response["sqlite_query"], con=engine)
            obj.data['df'] = df
            logger.info(f"Got data: {df}")
            query_succeeded = True

        except Exception as e:
            logger.error(
                f"Failed to run SQL query: {obj.response['sqlite_query']}"
            )

        logger.info(f"Running reference SQL query '{obj.data['sql']}'")
        df = pd.read_sql(obj.data["sql"], con=engine)
        logger.info(f"Got data: {df}")
        obj.data['reference_df'] = df

        logger.info(f"For question: {obj.data['question']}")
        logger.info(f"For query: {obj.response['sqlite_query']}")

        obj.data["query_succeeded"] = query_succeeded

    def preprocess(self, obj: PromptObject):
        new_prompt = make_llama_3_prompt(**self.make_prompt(obj.data))
        obj.prompt = new_prompt

    def make_prompt(self, data: dict):
        system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
        system += "Consider the nba_roster table with the following schema:\n"
        system += get_schema() + "\n"
        system += (
            "Write a sqlite SQL query that would help you answer the following question. Make sure each query ends with a semicolon:\n"
        )
        user = data["question"]
        return {
            "user": user,
            "system": system,
        }
    
class ScoreStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super().generate(
            prompt,
            output_type={"explanation": "str", "similar": ["true", "false"]},
            *args,
            **kwargs,
        )
        return results

    def preprocess(self, obj: PromptObject):
        obj.prompt = make_llama_3_prompt(**self.make_prompt(obj))
        logger.info(f"Scoring Stage Prompt:\n{obj.prompt}")

    def postprocess(self, obj: PromptObject):
        obj.data['is_matching'] = self.is_matching(obj.data, obj.response)
        obj.data['explanation'] = obj.response["explanation"]
        obj.data['similar'] = obj.response["similar"] == "true"

    def is_matching(self, data, response):
        return (str(data.get('df',"None")).lower() == str(data['reference_df']).lower() 
                or response['similar'] == "true")

    def make_prompt(self, obj: PromptObject):
        # Your evaluation model compares SQL output from the generated and reference SQL queries, using another LLM in the pipeline
        '''
        Note:
        Prompt tuning is important! 
        A previous iteration of this scoring pipeline said `Compare the following two dataframes to see if they are identical`.
        That prompt turned out to be too stringent of criteria.
        '''
        system_prompt = "Compare the following two dataframes. They are similar if they are almost identical, or if they convey the same information about the nba_roster dataset"
        system_prompt += "Respond with valid JSON {'explanation' : str, 'similar' : bool}"
        user_prompt = (
            f"========== Dataframe 1 =========\n{str(obj.data.get('df','None')).lower()}\n\n"
        )
        user_prompt += (
            f"========== Dataframe 2 =========\n{str(obj.data['reference_df']).lower()}\n\n"
        )
        user_prompt += f"Can you tell me if these dataframes are similar?"
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
async def run_eval(dataset, args):

    results = await run_evaluation_pipeline(dataset, args)

    print("Total results:", len(results))

    return results


async def run_evaluation_pipeline(dataset, args):
    results = EvaluationPipeline(args).call(dataset)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()
    return result_list


class EvaluationPipeline(GenerationPipeline):
    def __init__(self, args):
        super().__init__()
        self.query_stage = QueryStage(args.sql_model_name)
        self.score_stage = ScoreStage()


    def forward(self, x):
        x = self.query_stage(x)
        x = self.score_stage(x)
        return x
    
def load_gold_dataset(args):
    path = f"data/{args.gold_file_name}"

    with jsonlines.open(path) as reader:
        for index, obj in enumerate(reversed(list(reader))):
            if index >= args.max_examples:
                break
            yield PromptObject(prompt="", data=obj)

def save_eval_results(results, args):
    base_path = "./data/results"
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = f"nba_sql_pipeline_{now}"
    experiment_dir = os.path.join(base_path, experiment_name)
    os.makedirs(os.path.join(base_path, experiment_name))

    # Write args to file
    args_file_name = f"{experiment_dir}/args.txt"
    with open(args_file_name, "w") as writer:
        pprint(args.__dict__, writer)


    def is_correct(r):
        if (
            (result.data["query_succeeded"] and result.data['is_matching']) or 
            result.data["generated_query"] == result.data['sql']
        ):
            return True
        return False

    # Write sql results and errors to file
    results_file_name = f"{experiment_dir}/sql_results.jsonl"
    with jsonlines.open(results_file_name, "w") as writer:
        for result in results:
            if not is_correct(result):
                continue
            writer.write(
                {
                    "question": result.data['question'],
                    "query": result.data["generated_query"],
                    "query_succeeded": result.data["query_succeeded"],
                    "reference_sql": result.data['sql'],
                    "df": str(result.data.get('df', 'None')),
                    "reference_df": str(result.data['reference_df']),
                    'is_matching': result.data['is_matching'],
                    'similar': result.data['similar'],
                }
            )

    results_file_name = f"{experiment_dir}/sql_errors.jsonl"
    with jsonlines.open(results_file_name, "w") as writer:
        for result in results:
            if is_correct(result):
                continue
            writer.write(
                {
                    "question": result.data['question'],
                    "query": result.data["generated_query"],
                    "query_succeeded": result.data["query_succeeded"],
                    "df": str(result.data.get('df', 'None')),
                    "reference_df": str(result.data['reference_df']),
                    'is_matching': result.data['is_matching'],
                    'similar': result.data['similar'],
                }
            )

    # Write statistics to file
    average_sql_succeeded = sum(
        [result.data["query_succeeded"] for result in results]
    ) / len(results)
    average_correct = sum(
        [result.data["query_succeeded"] and result.data['is_matching'] for result in results]
    ) / len(results)

    file_name = f"{experiment_dir}/summary.txt"
    with open(file_name, "w") as writer:
        print(f"Total size of eval dataset: {len(results)}", file=writer)
        print(f"Total size of eval dataset: {len(results)}")
        print(f"Percent Valid SQL Syntax: {average_sql_succeeded*100}", file=writer)
        print(f"Percent Valid SQL Syntax: {average_sql_succeeded*100}")
        print(f"Percent Correct SQL Query: {average_correct*100}", file=writer)
        print(f"Percent Correct SQL Query: {average_correct*100}")




# Run the evaluation and you can see there is more valid SQL and correct queries.

# In[ ]:


args = Args(sql_model_name="a5ebf1c4879569101f32444afae5adcafbfce9c5a6ed13035fd892147f7d59bc")
dataset = load_gold_dataset(args)
results = await run_eval(dataset, args)
save_eval_results(results, args)


# ### Iteration 2
# Examine remaining errors.

# In[ ]:


get_ipython().system('cat sql_errors_example.jsonl')


# In[ ]:


get_ipython().system('cat "data/training_data/archive/generated_queries.jsonl" | grep "75th percentile"')


# In[ ]:


get_ipython().system('cat "data/training_data/archive/generated_queries_large.jsonl" | grep "75th percentile"')


# ### Filtering the dataset
# Next step is filtering. Manually create functions to filter the test set.

# In[ ]:


question_set = set()
sql_set = set()

def is_not_valid_sql(question, sql):
    try:
        df = pd.read_sql(sql, con=engine)
        return False
    except Exception as e:
        return True

def has_null_in_sql_or_question(question, sql):
    return "null" in sql.lower() or "null" in question

def returns_empty_dataframe(question, sql):
    try:
        df = pd.read_sql(sql, con=engine)
        return "Empty" in str(df) or "None" in str(df)
    except Exception as e:
        return False
    
def uses_avg_on_ht_column(question, sql):
    return "avg(ht)" in sql.lower() or "avg(salary" in sql.lower() 

filter_conditions = [is_not_valid_sql, has_null_in_sql_or_question, returns_empty_dataframe, uses_avg_on_ht_column]

def training_semicolon(sql):
    if sql.strip()[-1] != ";":
        return sql.strip() + ";"
    return sql

with jsonlines.open("data/training_data/archive/generated_queries_large.jsonl", "r") as reader:
    with jsonlines.open("data/training_data/generated_queries_large_filtered.jsonl", "w") as writer:
        for r in reader:
            if r["question"] in question_set or r["sql"] in sql_set:
                continue
            question_set.add(r["question"])
            sql_set.add(r["sql"])
            
            if any(c(r['question'], r['sql']) for c in filter_conditions):
                continue

            sql = training_semicolon(r['sql'])
            writer.write(
                {
                    "question": r["question"],
                    "sql": sql,
                }
            )


# Check the filtered dataset.

# In[ ]:


get_ipython().system('cat "data/training_data/archive/generated_queries_large_filtered.jsonl" | grep "75th percentile"')


# Manually clean the dataset. This has been done for you.

# In[ ]:


get_ipython().system('cat "data/training_data/archive/generated_queries_large_filtered_cleaned.jsonl" | grep "75th percentile"')


# Look at some other errors in the dataset.

# > The following cell is expected to create an error

# In[ ]:


df = pd.read_sql("SELECT AVG(CAST(SUBSTR(WT, 1, INSTR(WT,' ')) as INTEGER) FROM nba_roster WHERE WT!= 'NA') as median", con=engine)


# In[ ]:


get_ipython().system('cat "data/training_data/archive/generated_queries.jsonl" | grep "median weight"')


# In[ ]:


df = pd.read_sql("SELECT COLLEGE, COUNT(*) as count FROM nba_roster WHERE COLLEGE!= '--' GROUP BY COLLEGE ORDER BY count DESC LIMIT 1", con=engine)
print(df)


# Add more examples of median weight queries. (Done for you).

# In[ ]:


get_ipython().system('cat "data/training_data/archive/generated_queries_large_filtered_cleaned.jsonl" | grep "median weight"')


# In[ ]:


get_ipython().system('cat "data/training_data/archive/generated_queries_large_filtered_cleaned.jsonl" | grep "median"')


# In[ ]:


# Model tuned on `archive/generated_queries_large_filtered_cleaned.jsonl`
llm = lamini.Lamini(model_name="63fd73a775daf24216b46c680a1e963a8d1e02b21bca43fcea6c26737d2e887e")


# In[ ]:


question = """What is the median age of the Chicago Bulls?"""
system = f"""You are an NBA analyst with 15 years of experience writing complex SQL queries. Consider the nba_roster table with the following schema:
{get_schema()}

Write a sqlite query to answer the following question. Follow instructions exactly"""
prompt = make_llama_3_prompt(question, system)
print("Question:\n", question)

print("Answer:")
sql = llm.generate(prompt, max_new_tokens=200)
print(sql)


# In[ ]:


df = pd.read_sql(sql, con=engine)
print(df)


# Here is a larger pre-prepared dataset. 

# In[ ]:


get_ipython().system('cat data/gold-test-set-v2.jsonl')


# In[ ]:


args = Args(training_file_name="archive/generated_queries_v2_large_filtered_cleaned.jsonl")


# In[ ]:


llm = lamini.Lamini(model_name="meta-llama/Meta-Llama-3-8B-Instruct")


# In[ ]:


dataset = get_dataset(args, make_question)
finetune_args = get_default_finetune_args()


# This fine tuning step takes about 30 mintues to complete. The dispatch to run on the Lamini services is commented out and the pre-computed final results of the run are provided below. You can uncomment and run if you have modified data on your own.

# In[ ]:


#llm.train(
#    data_or_dataset_id=dataset,
#    finetune_args=finetune_args,
#    is_public=True,  # For sharing
#)


# Run eval platform again from lab 3.

# In[ ]:


# Collapsible or utils from Lesson 3 Lab for evaluation
class QueryStage(GenerationNode):
    def __init__(self, model_name):
        super().__init__(
            model_name=model_name,
            max_new_tokens=300,
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super().generate(
            prompt,
            output_type={"sqlite_query": "str"},
            *args,
            **kwargs,
        )
        return results


    def postprocess(self, obj: PromptObject):
        # Run both the generated and reference (Gold Dataset) SQL queries
        # Assessing whether the SQL queries succeeded in hitting the database (not correctness yet!)
        
        query_succeeded = False

        try:
            logger.info(f"Running SQL query '{obj.response['sqlite_query']}'")
            obj.data["generated_query"] = obj.response["sqlite_query"]
            df = pd.read_sql(obj.response["sqlite_query"], con=engine)
            obj.data['df'] = df
            logger.info(f"Got data: {df}")
            query_succeeded = True

        except Exception as e:
            logger.error(
                f"Failed to run SQL query: {obj.response['sqlite_query']}"
            )

        logger.info(f"Running reference SQL query '{obj.data['sql']}'")
        df = pd.read_sql(obj.data["sql"], con=engine)
        logger.info(f"Got data: {df}")
        obj.data['reference_df'] = df

        logger.info(f"For question: {obj.data['question']}")
        logger.info(f"For query: {obj.response['sqlite_query']}")

        obj.data["query_succeeded"] = query_succeeded

    def preprocess(self, obj: PromptObject):
        new_prompt = make_llama_3_prompt(**self.make_prompt(obj.data))
        obj.prompt = new_prompt

    def make_prompt(self, data: dict):
        system = "You are an NBA analyst with 15 years of experience writing complex SQL queries.\n"
        system += "Consider the nba_roster table with the following schema:\n"
        system += get_schema() + "\n"
        system += (
            "Write a sqlite SQL query that would help you answer the following question:\n"#"Write a sqlite SQL query that would help you answer the following question:\n"
        )
        user = data["question"]
        return {
            "user": user,
            "system": system,
        }
    
class ScoreStage(GenerationNode):
    def __init__(self):
        super().__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct",
            max_new_tokens=150,
        )

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        results = super().generate(
            prompt,
            output_type={"explanation": "str", "similar": ["true", "false"]},
            *args,
            **kwargs,
        )
        return results

    def preprocess(self, obj: PromptObject):
        obj.prompt = make_llama_3_prompt(**self.make_prompt(obj))
        logger.info(f"Scoring Stage Prompt:\n{obj.prompt}")

    def postprocess(self, obj: PromptObject):
        obj.data['is_matching'] = self.is_matching(obj.data, obj.response)
        obj.data['explanation'] = obj.response["explanation"]
        obj.data['similar'] = obj.response["similar"] == "true"

    def is_matching(self, data, response):
        return (str(data.get('df',"None")).lower() == str(data['reference_df']).lower() 
                or response['similar'] == "true")

    def make_prompt(self, obj: PromptObject):
        # Your evaluation model compares SQL output from the generated and reference SQL queries, using another LLM in the pipeline
        '''
        Note:
        Prompt tuning is important! 
        A previous iteration of this scoring pipeline said `Compare the following two dataframes to see if they are identical`.
        That prompt turned out to be too stringent of criteria.
        '''
        system_prompt = "Compare the following two dataframes. They are similar if they are almost identical, or if they convey the same information about the nba_roster dataset"
        system_prompt += "Respond with valid JSON {'explanation' : str, 'similar' : bool}"
        user_prompt = (
            f"========== Dataframe 1 =========\n{str(obj.data.get('df','None')).lower()}\n\n"
        )
        user_prompt += (
            f"========== Dataframe 2 =========\n{str(obj.data['reference_df']).lower()}\n\n"
        )
        user_prompt += f"Can you tell me if these dataframes are similar?"
        return {
            "system": system_prompt,
            "user": user_prompt
        }
    
async def run_eval(dataset, args):

    results = await run_evaluation_pipeline(dataset, args)

    print("Total results:", len(results))

    return results


async def run_evaluation_pipeline(dataset, args):
    results = EvaluationPipeline(args).call(dataset)

    result_list = []

    pbar = tqdm(desc="Saving results", unit=" results")
    async for result in results:
        result_list.append(result)
        pbar.update()
    return result_list


class EvaluationPipeline(GenerationPipeline):
    def __init__(self, args):
        super().__init__()
        self.query_stage = QueryStage(args.sql_model_name)
        self.score_stage = ScoreStage()


    def forward(self, x):
        x = self.query_stage(x)
        x = self.score_stage(x)
        return x
    
def load_gold_dataset(args):
    path = f"data/{args.gold_file_name}"

    with jsonlines.open(path) as reader:
        for index, obj in enumerate(reversed(list(reader))):
            if index >= args.max_examples:
                break
            yield PromptObject(prompt="", data=obj)

def save_eval_results(results, args):
    base_path = "./data/results"
    now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    experiment_name = f"nba_sql_pipeline_{now}"
    experiment_dir = os.path.join(base_path, experiment_name)
    os.makedirs(os.path.join(base_path, experiment_name))

    # Write args to file
    args_file_name = f"{experiment_dir}/args.txt"
    with open(args_file_name, "w") as writer:
        pprint(args.__dict__, writer)


    def is_correct(r):
        if (
            (result.data["query_succeeded"] and result.data['is_matching']) or 
            result.data["generated_query"] == result.data['sql']
        ):
            return True
        return False

    # Write sql results and errors to file
    results_file_name = f"{experiment_dir}/sql_results.jsonl"
    with jsonlines.open(results_file_name, "w") as writer:
        for result in results:
            if not is_correct(result):
                continue
            writer.write(
                {
                    "question": result.data['question'],
                    "query": result.data["generated_query"],
                    "query_succeeded": result.data["query_succeeded"],
                    "reference_sql": result.data['sql'],
                    "df": str(result.data.get('df', 'None')),
                    "reference_df": str(result.data['reference_df']),
                    'is_matching': result.data['is_matching'],
                    'similar': result.data['similar'],
                }
            )

    results_file_name = f"{experiment_dir}/sql_errors.jsonl"
    with jsonlines.open(results_file_name, "w") as writer:
        for result in results:
            if is_correct(result):
                continue
            writer.write(
                {
                    "question": result.data['question'],
                    "query": result.data["generated_query"],
                    "query_succeeded": result.data["query_succeeded"],
                    "df": str(result.data.get('df', 'None')),
                    "reference_df": str(result.data['reference_df']),
                    'is_matching': result.data['is_matching'],
                    'similar': result.data['similar'],
                }
            )

    # Write statistics to file
    average_sql_succeeded = sum(
        [result.data["query_succeeded"] for result in results]
    ) / len(results)
    average_correct = sum(
        [result.data["query_succeeded"] and result.data['is_matching'] for result in results]
    ) / len(results)

    file_name = f"{experiment_dir}/summary.txt"
    with open(file_name, "w") as writer:
        print(f"Total size of eval dataset: {len(results)}", file=writer)
        print(f"Total size of eval dataset: {len(results)}")
        print(f"Percent Valid SQL Syntax: {average_sql_succeeded*100}", file=writer)
        print(f"Percent Valid SQL Syntax: {average_sql_succeeded*100}")
        print(f"Percent Correct SQL Query: {average_correct*100}", file=writer)
        print(f"Percent Correct SQL Query: {average_correct*100}")




# Use pretrained model trained with the above dataset.

# In[ ]:


args = Args(sql_model_name="3f7e740c0ea2227631a30d293b51564ad1b80727c3768a3b136fbae93170c1e2", gold_file_name='gold-test-set-v2.jsonl')
dataset = load_gold_dataset(args)
results = await run_eval(dataset, args)
save_eval_results(results, args)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




