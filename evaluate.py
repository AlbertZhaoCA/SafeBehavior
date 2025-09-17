from libs.evalaute.evaluator import Evaluator
from libs.defenders.registry import load_defender
from utils.data import parse_unknown_args
from helpers.dataset import get_attack_prompts
from libs.defenders import *
from config import EVALUATION_MODEL_ENDPOINT,EVALUATION_MODEL_KEY,EVALUATION_MODEL,SUBSCRIPTION_KEY,ENDPOINT

def getArgs():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate")
    parser.add_argument("--mode", type=str, default="jailbreak", help="mode of evaluation, either jailbreak or refusal")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="model used for evaluation")
    parser.add_argument("--model_type", type=str, default="remote", help="model type used for evaluation")
    parser.add_argument("--model_endpoint", type=str, default=EVALUATION_MODEL_ENDPOINT, help="model endpoint used for evaluation")
    parser.add_argument("--evaluate_model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="model used for evaluation")
    parser.add_argument("--evaluate_model_type", type=str, default="remote", help="model type used for evaluation")
    parser.add_argument("--evaluate_model_endpoint", type=str, help="model type used for evaluation")
    parser.add_argument("--evaluate_model_key", type=str, help="model type used for evaluation")
    parser.add_argument("--defender", type=str, default="ppl_calculator", help="model used for defense")
    parser.add_argument("--attacker", type=str, default="gcg", help="model used for attack")
    parser.add_argument("--dataset", type=str, default="data/advbench/harmful_behaviors.csv", help="Dataset used for evaluation")
    parser.add_argument("--path", type=str, default="csv", help="Dataset path to save the results")
    parser.add_argument("--attack_dataset", type=str, default="data/gcg/qwen.csv", help="Dataset used for attack")
    parser.add_argument("--split", type=str, default="train", help="split")
    parser.add_argument("--repo_name", type=str, default=None, help="repo to save the result")
    parser.add_argument("--exp", type=str, default=None, help="running experiment name")

    args, unknown_args = parser.parse_known_args()
    return args, unknown_args

args,unknown_args =  getArgs()
unknown_args_dict = parse_unknown_args(unknown_args)
metadata = {
    "attacker": args.attacker,
    "defender": args.defender,
    "model": args.model,
    "dataset": args.attack_dataset,
    "exp": args.exp,
}

evaluation_model = EVALUATION_MODEL if args.evaluate_model is None else args.evaluate_model
evaluation_model_key = EVALUATION_MODEL_KEY if args.evaluate_model_key is None else args.evaluate_model_key
evaluation_model_endpoint = EVALUATION_MODEL_ENDPOINT if args.evaluate_model_endpoint is None else args.evaluate_model_endpoint
model_endpoint = args.model_endpoint if args.model_endpoint is not None else evaluation_model_endpoint
attack_dataset = get_attack_prompts(args.attack_dataset, split=args.split, attacker=args.attacker)
defender = load_defender(args.defender,args.model,type=args.model_type,base_url=ENDPOINT,key=SUBSCRIPTION_KEY,**unknown_args_dict)
evaluator = Evaluator(metadata=metadata,llm=evaluation_model,type=args.evaluate_model_type,base_url=evaluation_model_endpoint, key=evaluation_model_key,mode=args.mode)
evaluator.evaluate(attack_dataset, defender)

