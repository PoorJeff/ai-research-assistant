from pathlib import Path
import argparse
import sys

def main() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from src.real_paper_evaluation import run_real_paper_evaluation, write_evaluation_artifacts

    parser = argparse.ArgumentParser(description="Run real arXiv PDF retrieval evaluation.")
    parser.add_argument(
        "--embedding-backend",
        choices=["sentence-transformer", "hash"],
        default="sentence-transformer",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/papers/real_demo",
        help="Directory for downloaded PDFs. This path is ignored by Git.",
    )
    args = parser.parse_args()

    result = run_real_paper_evaluation(
        pdf_dir=Path(args.pdf_dir),
        embedding_backend=args.embedding_backend,
    )
    write_evaluation_artifacts(result)
    print(f"Average Recall@3: {result.average_recall_at_3:.2f}")
    print(f"Average Recall@5: {result.average_recall_at_5:.2f}")
    print(f"Citation coverage: {result.citation_coverage_count}/{len(result.query_results)}")


if __name__ == "__main__":
    main()
