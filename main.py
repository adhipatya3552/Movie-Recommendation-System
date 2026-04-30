# Import functions from your modules
from src.collaborative_filtering import recommend_movies
from src.content_based import recommend_by_content
from src.hybrid_model import hybrid_recommendation


def main():
    print("🎬 Movie Recommendation System")
    print("----------------------------------")

    while True:
        print("\nChoose Recommendation Type:")
        print("1. Collaborative Filtering")
        print("2. Content-Based Filtering")
        print("3. Hybrid Recommendation")
        print("4. Exit")

        choice = input("Enter your choice (1-4): ")

        if choice == "1":
            movie_title = input("Enter Movie Title: ")
            recommendations = recommend_movies(movie_title)

            print("\nRecommended Movies:")
            for movie in recommendations:
                print(movie)

        elif choice == "2":
            movie_title = input("Enter Movie Title: ")
            recommendations = recommend_by_content(movie_title)

            print("\nRecommended Movies:")
            for movie in recommendations:
                print(movie)

        elif choice == "3":
            movie_title = input("Enter Movie Title: ")
            recommendations = hybrid_recommendation(movie_title)

            print("\nRecommended Movies:")
            for movie in recommendations:
                print(movie)

        elif choice == "4":
            print("Exiting... 👋")
            break

        else:
            print("Invalid choice. Please try again.")


if __name__ == "__main__":
    main()