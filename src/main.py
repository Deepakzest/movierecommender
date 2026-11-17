from recommender import recommend_movies

print("=" * 45)
print("       MOVIE RECOMMENDATION SYSTEM")
print("=" * 45)

while True:

    print("\n1. Get Movie Recommendations")
    print("2. Exit")

    choice = input("\nEnter your choice: ")

    if choice == "1":

        user_id = int(input("\nEnter User ID (1-943): "))

        if user_id < 1 or user_id > 943:
            print("Invalid User ID!")
            continue

        recommend_movies(user_id)

    elif choice == "2":

        print("\nThank you for using Movie Recommendation System!")
        break

    else:
        print("Invalid choice!")