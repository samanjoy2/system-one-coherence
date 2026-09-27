import argparse


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="command",required=True)
    sub.add_parser("prepare-data")
    r = sub.add_parser("run")
    r.add_argument("--model",choices=["jev","laya"],required=True)
    r.add_argument("--limit",type=int)
    r.add_argument("--permutation",type=int,default=0)
    r.add_argument("--repeat",type=int,default=0)
    r.add_argument("--precision",choices=["int8","fp32"],default="int8")
    r.add_argument("--audit",action="store_true")
    analysis_parser = sub.add_parser("analyze")
    analysis_parser.add_argument("--preview",action="store_true")
    a = sub.add_parser("audit")
    a.add_argument("--model",choices=["jev","laya"],required=True)
    a.add_argument("--precision",action="store_true")
    args = p.parse_args()
    if args.command == "prepare-data":
        from .prepare import prepare_data
        prepare_data()
    elif args.command == "run":
        from .run import run
        run(args.model,args.limit,args.permutation,args.repeat,args.precision,args.audit)
    elif args.command == "analyze":
        from .analysis import analyze
        analyze(args.preview)
    elif args.command == "audit":
        from .audit import run_audit
        run_audit(args.model,args.precision)


if __name__ == "__main__": main()
