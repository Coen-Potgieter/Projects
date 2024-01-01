import numpy as np


ITEMS = ["bicycle",
         "bread",
         "pizza",
         "windmill",
         "tree",
         "cloud",
         "fork",
         "sun",
         "line",
         "leaf",
         "ocean",
         "pencil",
         "star",
         "stethoscope"]


def get_data():

    # lis_of_npArrays = [
    #     np.load(f"Assets/data/{elem}.npy").T[:, 0:6000] / 255 for elem in ITEMS]
    lis_of_npArrays = [
        np.load(f"Assets/data/{elem}.npy").T[:, 0:50_000] / 255 for elem in ITEMS]

    running_sum = 0
    for idx, elem in enumerate(lis_of_npArrays):
        running_sum += elem.shape[1]
        print(f"{ITEMS[idx]} -> {elem.shape[1]} Ex.")

    print(f"With Total of {running_sum} Ex.")


def main():
    get_data()


if __name__ == "__main__":
    main()
