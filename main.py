from args import parser_args, run_args

# Collects the flags that where ran
# Runs the flags to choose which workflows to use
def main():
    parser = parser_args()
    run_args(parser)
    return

if __name__ == "__main__":
    main()
    
