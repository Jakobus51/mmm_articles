from pandas import date_range, DataFrame
from random import uniform, randint, seed


class FakeMarketing:
    def __init__(self, random_state: int):
        seed(random_state)
        # create dates for 3 years
        dates = date_range(start="1/1/2021", end="12/31/2023")
        data = {
            "Date": [],
            "Channel_1": [],
            "Channel_2": [],
            "Channel_3": [],
            "Channel_4": [],
            "Channel_5": [],
            "Channel_6": [],
            "Channel_7": [],
            "Channel_8": [],
            "Promo_1": [],
            "Promo_2": [],
        }

        # generate fake marketing expenditures and promo days
        for date in dates:
            data["Date"].append(date)
            data["Channel_1"].append(round(uniform(10, 1000), 2))
            data["Channel_2"].append(round(uniform(10, 5000), 2))
            data["Channel_3"].append(round(uniform(1000, 1500), 2))
            data["Channel_4"].append(round(uniform(100, 200), 2))
            data["Channel_5"].append(round(uniform(1000, 1500), 2))
            data["Channel_6"].append(round(uniform(100, 2500), 2))
            data["Channel_7"].append(round(uniform(100, 2500), 2))
            data["Channel_8"].append(round(uniform(1000, 2500), 2))

            # Check if the date is Black Friday (the day after the fourth Thursday of November)
            if (date.month == 11) and (date.weekday() == 4) and (20 <= date.day <= 27):
                data["Promo_1"].append(1)
            else:
                data["Promo_1"].append(0)

            # Check if the date is the first of August or February
            if (date.month in [2, 8]) and (date.day == 1):
                data["Promo_2"].append(1)
            else:
                data["Promo_2"].append(0)

        # fill some entries of the channels with zeros
        data["Channel_1"] = self._set_zeros(data["Channel_1"], 4)
        data["Channel_2"] = self._set_zeros(data["Channel_2"], 8)
        data["Channel_3"] = self._set_zeros(data["Channel_3"], 2)
        data["Channel_4"] = self._set_zeros(data["Channel_4"], 4)
        data["Channel_5"] = self._set_zeros(data["Channel_5"], 2)
        data["Channel_6"] = self._set_zeros(data["Channel_6"], 9)
        data["Channel_7"] = self._set_zeros(data["Channel_7"], 15)
        data["Channel_8"] = self._set_zeros(data["Channel_8"], 9)

        # Save to dataframe and then to csv
        df = DataFrame(data)
        # df.to_csv("src/data_generation/fake_marketing.csv", index=False)
        self.df = df

    def _set_zeros(self, lst: list, times):
        """Randomly fill the given list with n times 10 zeros"""
        # Randomly choose a start index for the sequence to replace. Ensure it's at least 10 away from the end of the list.
        for _ in range(times):
            start_index = randint(0, len(lst) - 10)

            # Replace 10 consecutive elements starting at the chosen index with 0
            lst[start_index : start_index + 10] = [0] * 10
        return lst


if __name__ == "__main__":
    FakeMarketing()
