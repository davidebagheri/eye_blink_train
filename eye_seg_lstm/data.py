import numpy as np
import os
import csv
from math import floor
from sklearn.utils import shuffle

class CSVDataReader:
    """
    Transform the landmark from csv file from string to np.array
    """

    def _get_landmark(self, s):
        i = s.find(",")
        x = int(s[1:i])
        y = int(s[i + 1:len(s) - 1])
        return np.array([x, y])

    """
    Get landmark positions in normalized frame
    """

    def _get_landmark_position(self, global_landmarks):
        center_snx = (global_landmarks[0] + global_landmarks[3]) / 2
        center_dxt = (global_landmarks[6] + global_landmarks[9]) / 2

        offset = np.linalg.norm(global_landmarks[3] - global_landmarks[6])

        # Origin of the normalized frame for each eye
        ur_snx = np.array([int(center_snx[0] + offset / 1.5), int(center_snx[1] - offset / 3)])
        ul_dxt = np.array([int(center_dxt[0] - offset / 1.5), int(center_dxt[1] - offset / 3)])

        # Compute coordinates for the left eye landmarks
        l_ldmk = [np.array([-1 / (2 * offset / 1.5), 1 / (2 * offset / 3)]) * (snx_landmark - ur_snx) for snx_landmark
                  in global_landmarks[:6]]

        # Compute coordinates for the right eye landmarks
        r_ldmk = [np.array([1 / (2 * offset / 1.5), 1 / (2 * offset / 3)]) * (dxt_landmark - ul_dxt) for dxt_landmark in
                  global_landmarks[6:]]

        return [[l_ldmk[3], l_ldmk[2], l_ldmk[1], l_ldmk[0], l_ldmk[5], l_ldmk[4]], r_ldmk]

    """
    Read data from csv file: if the measure is repeated for left and right eyes,
    the left one is stored as first, the right as second
    """

    def read_csv(self, file_path):
        f = open(file_path, 'r')
        reader = csv.reader(f, delimiter=';')

        landmarks_timeline = []
        ears_timeline = []
        eye_status_timeline = []
        pupil_area_timeline = []
        white_area_timeline = []
        background_area_timeline = []

        for row in reader:
            # First row
            if row[0] == 'id':
                continue

            # Eyes landmarks
            landmarks = [self._get_landmark(s) for s in row[37:49]]
            landmarks_timeline.append(landmarks)

            # Ear measurements
            ears_timeline.append([float(row[69]), float(row[70])])

            # Eye status
            eye_status_timeline.append(int(row[71]))

            # Pupil area
            pupil_area_timeline.append([float(row[72]), float(row[75])])

            # White area
            white_area_timeline.append([float(row[73]), float(row[76])])

            # Background area
            background_area_timeline.append([float(row[74]), float(row[77])])

        return eye_status_timeline, landmarks_timeline, ears_timeline, pupil_area_timeline, white_area_timeline, background_area_timeline

    """
    Get samples intervals from eye status timeline
    """

    def get_sample_intervals_from_csv_data(self, eye_status_timeline, sample_length):
        def get_interval_of_ones(idx):
            # Get indexes of the occurrences of ones
            indices = [i + idx for i, x in enumerate(eye_status_timeline[idx:idx + sample_length]) if x == 1]

            # Avoid double intervals
            for i in range(len(indices) - 1):
                if indices[i + 1] - indices[i] > 1:
                    return (-1, -1)

            # Get the sample centered on the interval
            medium_idx = int((indices[0] + indices[-1]) / 2)
            low = medium_idx - floor((sample_length - 1) / 2)
            high = low + sample_length - 1

            # Recheck for double intervalsof ones
            indices = [i for i, x in enumerate(eye_status_timeline[low:high]) if x == 1]

            for i in range(len(indices) - 1):
                if indices[i + 1] - indices[i] > 1:
                    return (-1, -1)

            return [low, high]

        unblink = []
        blink = []

        i = 0

        while i < len(eye_status_timeline) - sample_length:
            # Blink found
            if 1 in eye_status_timeline[i:i + sample_length]:
                low, high = get_interval_of_ones(i)

                # Good sample found
                if (low > -1 and high > -1):
                    blink.append([low, high])
                    i = high + 1
                    continue

            # Unblink found
            elif (1 not in eye_status_timeline[i - min(i, 3):i]) and (1 not in eye_status_timeline[
                                                                               i + sample_length:min(
                                                                                       len(eye_status_timeline),
                                                                                       i + sample_length + 3)]):
                unblink.append([i, i + sample_length - 1])

            i += sample_length

        return [blink, unblink]

    """
    Read data from csv file and returns the numpy tensor with dimensions (n_samples, measures, time_steps),
    where measures are (in order):
        0) ear
        1) pupil
        2) white
        3) backgroud
        4) landmark_0_x
        5) landmark_0_y
        6) landmark_1_x
        7) landmark_1_y
        8) landmark_2_x
        9) landmark_2_y
        10) landmark_3_x
        11) landmark_3_y
        12) landmark_4_x
        13) landmark_4_y
        14) landmark_5_x
        15) landmark_5_y

        Landmark definition:


                      X <--     --> X                    Y down
             2 --- 1                   1 --- 2 
          3<         > 0           0 <         > 3
             4 --- 5        | \        5 --- 4

    """

    def get_sample_from_csv(self, file_path, sample_length):
        # Read data
        data = self.read_csv(file_path)

        # Get intervals
        blink_intervals, unblink_intervals = self.get_sample_intervals_from_csv_data(data[0], sample_length)

        n_blinks = len(blink_intervals)
        n_unblinks = len(unblink_intervals)

        # Build data frame
        X = np.empty(shape=((n_blinks + n_unblinks) * 2, 16, sample_length))
        Y = np.empty(shape=((n_blinks + n_unblinks) * 2), dtype=np.uint8)

        for i in range(n_unblinks):
            landmark_sample = np.array(data[1][unblink_intervals[i][0]:unblink_intervals[i][1] + 1])
            ears_sample = np.array(data[2][unblink_intervals[i][0]:unblink_intervals[i][1] + 1])
            pupil_area_sample = np.array(data[3][unblink_intervals[i][0]:unblink_intervals[i][1] + 1])
            white_area_sample = np.array(data[4][unblink_intervals[i][0]:unblink_intervals[i][1] + 1])
            background_area_sample = np.array(data[5][unblink_intervals[i][0]:unblink_intervals[i][1] + 1])

            X[i * 2:(i + 1) * 2, 0] = ears_sample.T
            X[i * 2:(i + 1) * 2, 1] = pupil_area_sample.T
            X[i * 2:(i + 1) * 2, 2] = white_area_sample.T
            X[i * 2:(i + 1) * 2, 3] = background_area_sample.T
            Y[i * 2:(i + 1) * 2] = [0, 0]

            for j in range(len(landmark_sample)):
                landmark_pos = self._get_landmark_position(landmark_sample[j])
                X[i * 2, 4:, j] = np.concatenate(landmark_pos[0], axis=0)
                X[i * 2 + 1, 4:, j] = np.concatenate(landmark_pos[1], axis=0)

        for i in range(n_blinks):
            landmark_sample = np.array(data[1][blink_intervals[i][0]:blink_intervals[i][1] + 1])
            ears_sample = np.array(data[2][blink_intervals[i][0]:blink_intervals[i][1] + 1])
            pupil_area_sample = np.array(data[3][blink_intervals[i][0]:blink_intervals[i][1] + 1])
            white_area_sample = np.array(data[4][blink_intervals[i][0]:blink_intervals[i][1] + 1])
            background_area_sample = np.array(data[5][blink_intervals[i][0]:blink_intervals[i][1] + 1])

            X[n_unblinks * 2 + i * 2:n_unblinks * 2 + (i + 1) * 2, 0] = ears_sample.T
            X[n_unblinks * 2 + i * 2:n_unblinks * 2 + (i + 1) * 2, 1] = pupil_area_sample.T
            X[n_unblinks * 2 + i * 2:n_unblinks * 2 + (i + 1) * 2, 2] = white_area_sample.T
            X[n_unblinks * 2 + i * 2:n_unblinks * 2 + (i + 1) * 2, 3] = background_area_sample.T
            Y[n_unblinks * 2 + i * 2:n_unblinks * 2 + (i + 1) * 2] = [1, 1]

            for j in range(len(landmark_sample)):
                landmark_pos = self._get_landmark_position(landmark_sample[j])
                X[n_unblinks * 2 + i * 2, 4:, j] = np.concatenate(landmark_pos[0], axis=0)
                X[n_unblinks * 2 + i * 2 + 1, 4:, j] = np.concatenate(landmark_pos[1], axis=0)

        return X, Y

    def get_dataset_from_csv(self, dataset_path, sample_length, shuffle_data=True, balance_data=True):
        folders = [os.path.join(dataset_path, folder, "landmarks.csv") for folder in os.listdir(dataset_path)]

        if len(folders) == 0:
            print("no samples in folder {}".format(dataset_path))
            return

        X, Y = self.get_sample_from_csv(folders[0], sample_length)

        for i in range(1, len(folders)):
            X_sample, Y_sample = self.get_sample_from_csv(folders[i], sample_length)
            X = np.concatenate((X, X_sample), axis=0)
            Y = np.concatenate((Y, Y_sample), axis=0)

        n_blinks = Y.tolist().count(1)
        n_unblinks = Y.tolist().count(0)
        if shuffle_data:
            X, Y = shuffle(X, Y)

        if balance_data:
            to_delete = n_blinks > n_unblinks
            indexes_to_delete = [i for i, x in enumerate(Y) if x == to_delete][:np.abs(n_unblinks - n_blinks)]
            indexes_to_delete = shuffle(indexes_to_delete)
            mask = np.ones(len(Y), dtype=bool)
            mask[indexes_to_delete] = False
            X = X[mask]
            Y = Y[mask]

        print("Read {} blinks and {} unblinks".format(n_blinks, n_unblinks))
        return X, Y