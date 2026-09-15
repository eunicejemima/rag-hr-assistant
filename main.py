from dataprocessor import process_documents
from retriever import Retriever
from generator import generate_answer


def build_vector_store():

    print("\nProcessing HR documents...\n")

    chunks = process_documents()

    retriever = Retriever()

    retriever.create_vector_store(
        chunks
    )

    return retriever


def main():

    print("=" * 50)

    print("       RAG HR ASSISTANT")

    print("=" * 50)


    retriever = Retriever()


    try:

        retriever.load_vector_store()

    except FileNotFoundError:

        print(
            "\nVector store not found."
        )

        print(
            "Creating vector store...\n"
        )

        retriever = build_vector_store()


    print("\nHR Assistant is ready!")

    print(
        "Ask questions about company policies."
    )

    print(
        "Type 'exit' to quit.\n"
    )


    while True:

        question = input(
            "You: "
        ).strip()


        if question.lower() == "exit":

            print(
                "\nGoodbye!"
            )

            break


        if not question:

            continue


        print(
            "\nSearching company documents..."
        )


        results = retriever.search(
            question,
            top_k=3
        )


        context_parts = []


        print("\nRelevant documents:")


        for result in results:

            print(
                f"- {result['source']}"
            )

            context_parts.append(
                f"Source: {result['source']}\n"
                f"{result['text']}"
            )


        context = "\n\n".join(
            context_parts
        )


        print(
            "\nGenerating answer..."
        )


        answer = generate_answer(
            question,
            context
        )


        print(
            "\nHR Assistant:"
        )

        print(answer)

        print("\n" + "-" * 50)


if __name__ == "__main__":

    main()