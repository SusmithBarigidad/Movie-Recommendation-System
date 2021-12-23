
import decimal
class user:

    def __init__(self, _id, ratingList, input_ratings, trainTestBool):

        self.sumSq = 1
        self.rmse = 1
        self.bool_train = trainTestBool
        self.mean = 1
        _id = str(_id)
        if self.bool_train == 0:
            self.id = _id
            self.ratings = {}
            
            for ix, rt in enumerate(ratingList):
                self.ratings[str(ix+1)] = int(rt)

        elif self.bool_train == 1:
            self.id = _id
            self.targets = [] 
            self.ratings = input_ratings 
            for key in input_ratings:
                if input_ratings[key] == 0:
                    self.targets.append(key)

    def measure_sumSq(self):
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                total = total + int(self.ratings[key])**2

        return decimal.Decimal(total).sqrt()

    def measure_rmse(self):
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                total = total + (int(self.ratings[key]) - self.meanRating)**2
                
        return decimal.Decimal(total).sqrt()

    def measure_meanRating(self):
        count = 0
        total = 0
        for key in self.ratings:
            if int(self.ratings[key]) != 0:
                count += 1
                total += int(self.ratings[key])

        return (total/count)

    def appendRating(self, movie_id, rating):
        self.ratings[movie_id] = int(rating)
        if rating == 0 and self.bool_train == 1:
            self.targets.append(movie_id)

    def get_rating(self, movie_id):

        if movie_id in self.ratings:
            return int(self.ratings[movie_id])

        return 0;