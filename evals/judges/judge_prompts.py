"""LLM-as-judge prompt templates for evaluating agent responses."""

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator for a fitness AI training coach.
You will be given:
1. A user query
2. Context about the user's situation
3. The AI agent's response
4. A list of evaluation criteria

Your job is to score the response on each criterion.

For EACH criterion, respond with:
- PASS if the response clearly satisfies the criterion
- FAIL if the response does not satisfy the criterion
- PARTIAL if the response partially addresses it but is incomplete

Then give an overall score from 0.0 to 1.0 where:
- 1.0 = all criteria fully met
- 0.0 = no criteria met

Respond ONLY in this exact JSON format:
{
  "criteria_results": [
    {"criterion": "...", "result": "PASS|FAIL|PARTIAL", "reasoning": "..."},
    ...
  ],
  "overall_score": 0.0,
  "summary": "Brief explanation of the overall assessment"
}"""


JUDGE_USER_TEMPLATE = """## User Query
{query}

## User Context
{context}

## Agent Response
{response}

## Evaluation Criteria
{criteria}

Evaluate the agent's response against each criterion. Respond in JSON only."""