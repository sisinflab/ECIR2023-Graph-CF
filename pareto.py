import pandas as pd
import numpy as np
import os


class ObjectivesSpace:
    def __init__(self, df, functions, path, model):
        self.functions = functions
        self.df = df[df.columns.intersection(self._constr_obj())]
        self.points = self._get_points()
        self.path = path[:-4]
        self.model = model

    def _constr_obj(self):
        objectives = list(self.functions.keys())
        objectives.insert(0, 'model')
        return objectives

    def _get_points(self):
        pts = self.df.to_numpy()
        # pts = obj_pts.copy()
        # obj_pts = obj_pts[obj_pts.sum(1).argsort()[::-1]]
        factors = np.array(list(map(lambda x: 1 if x == 'max' else -1, list(self.functions.values()))))
        pts[:, 1:] = pts[:, 1:] * factors
        # sort points by decreasing sum of coordinates: the point having the greatest sum will be non dominated
        pts = pts[pts[:, 1:].sum(1).argsort()[::-1]]
        # initialize a boolean mask for non dominated and dominated points (in order to be contrastive)
        non_dominated = np.ones(pts.shape[0], dtype=bool)
        dominated = np.zeros(pts.shape[0], dtype=bool)
        for i in range(pts.shape[0]):
            # process each point in turn
            n = pts.shape[0]
            # definition of Pareto optimality: for each point in the iteration, we find all points non dominated by
            # that point.
            mask1 = (pts[i + 1:, 1:] >= pts[i, 1:])
            mask2 = np.logical_not(pts[i + 1:, 1:] <= pts[i, 1:])
            non_dominated[i + 1:n] = (np.logical_and(mask1, mask2)).any(1)
            # A point could dominate another point, but it could also be dominated by a previous one in the iteration.
            # The following row take care of this situation by "keeping in memory" all dominated points in previous
            # iterations.
            dominated[i + 1:n] = np.logical_or(np.logical_not(non_dominated[i + 1:n]), dominated[i + 1:n])
        pts[:, 1:] = pts[:, 1:] * factors
        return pts[(np.logical_not(dominated))], pts[dominated]

    def get_nondominated(self):
        return self.points[0]

    def get_dominated(self):
        return self.points[1]

    def to_csv_new(self):
        df_nondominated = pd.DataFrame(self.points[0], columns=self._constr_obj())
        df_nondominated = df_nondominated.sort_values(by=list(self.functions.keys())[1])
        df_nondominated.to_csv('results_' + self.path + '_' + str(self._constr_obj()[1]) + '_' + self._constr_obj()[2] + '_nondominated.csv', index=False)


if __name__ == '__main__':
    files = os.listdir('daniele/men')
    for file in files:
        print(file)
        path = 'daniele/men/' + file
        model = pd.read_csv(path, sep='\t')
        obj = ObjectivesSpace(model, {'nDCG': 'max', 'APLT': 'max'}, path, file.split('.')[0])
        # obj = ObjectivesSpace(model, {'nDCG': 'max', 'UserMADranking_WarmColdUsers': 'min'}, path, file.split('.')[0])
        # obj = ObjectivesSpace(model, {'APLT': 'max', 'UserMADranking_WarmColdUsers': 'min'}, path, file.split('.')[0])
        obj.to_csv_new()
