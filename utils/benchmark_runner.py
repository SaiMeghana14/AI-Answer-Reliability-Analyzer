import pandas as pd

from utils.generator_basic import generate_basic_answer
from utils.retrieval_engine import RetrievalEngine
from utils.source_agreement import SourceAgreement
from utils.reliability_engine import ReliabilityEngine


class BenchmarkRunner:

    def __init__(self):

        self.engine = RetrievalEngine()

    def evaluate_question(self, question):

        baseline = generate_basic_answer(question)

        sources = self.engine.retrieve(question)

        evidence = SourceAgreement.merged_text(sources)

        retrieval = f"Based on retrieved sources:\n\n{evidence}"

        base_eval = ReliabilityEngine.evaluate(
            baseline,
            evidence
        )

        retrieval_eval = ReliabilityEngine.evaluate(
            retrieval,
            evidence
        )

        return {

            "Question": question,

            "Baseline Score":
                base_eval["final"],

            "Retrieval Score":
                retrieval_eval["final"],

            "Winner":
                "Retrieval"
                if retrieval_eval["final"] >
                   base_eval["final"]
                else "Baseline"

        }

    def run(self, csv_file):

        df = pd.read_csv(csv_file)

        rows=[]

        for _,row in df.iterrows():

            rows.append(

                self.evaluate_question(
                    row["question"]
                )

            )

        return pd.DataFrame(rows)
