import numpy as np


class StepFeatureExtractor(object):

    def extract(self,
                steps):
        features = np.zeros((len(steps), 29), dtype=float)
        for s_idx, step in enumerate(steps):

            energy = np.sqrt(np.power(step.data['ver'].values, 2) +
                             np.power(step.data['hor'].values, 2) +
                             np.power(step.data['fwd'].values, 2))

            # Duration
            duration = step.duration

            # Mean
            mean_ver = np.mean(step.data['ver'].values)
            mean_hor = np.mean(step.data['hor'].values)
            mean_fwd = np.mean(step.data['fwd'].values)
            mean_eng = np.mean(energy)

            # Variance
            var_ver = np.var(step.data['ver'].values)
            var_hor = np.var(step.data['hor'].values)
            var_fwd = np.var(step.data['fwd'].values)
            var_eng = np.var(energy)

            # Max
            max_ver = np.max(step.data['ver'].values)
            max_hor = np.max(step.data['hor'].values)
            max_fwd = np.max(step.data['fwd'].values)
            max_eng = np.max(energy)

            # Min
            min_ver = np.min(step.data['ver'].values)
            min_hor = np.min(step.data['hor'].values)
            min_fwd = np.min(step.data['fwd'].values)
            min_eng = np.min(energy)

            # Range
            range_ver = max_ver - min_ver
            range_hor = max_hor - min_hor
            range_fwd = max_fwd - min_fwd
            range_eng = max_eng - min_eng

            # Sum
            sum_ver = np.sum(step.data['ver'].values)
            sum_hor = np.sum(step.data['hor'].values)
            sum_fwd = np.sum(step.data['fwd'].values)
            sum_eng = np.sum(energy)

            # Max-mean
            mm_ver = max_ver - mean_ver
            mm_hor = max_hor - mean_hor
            mm_fwd = max_fwd - mean_fwd
            mm_eng = max_eng - mean_eng

            # Step amplitude
            sa_ver = mm_ver / float(step.size)
            sa_hor = mm_hor / float(step.size)
            sa_fwd = mm_fwd / float(step.size)
            sa_eng = mm_eng / float(energy.size)

            # Root mean square
            rms_ver = np.sqrt(np.sum(np.power(step.data['ver'].values, 2)) / float(step.size))
            rms_hor = np.sqrt(np.sum(np.power(step.data['hor'].values, 2)) / float(step.size))
            rms_fwd = np.sqrt(np.sum(np.power(step.data['fwd'].values, 2)) / float(step.size))
            rms_eng = np.sqrt(np.sum(np.power(energy, 2)) / float(energy.size))

            # # Pearson's Correlation
            # coeff_ver_hor, pval_ver_hor = pearsonr(acc.ver, acc.hor)
            # coeff_ver_fwd, pval_ver_fwd = pearsonr(acc.ver, acc.fwd)
            # coeff_hor_fwd, pval_hor_fwd = pearsonr(acc.hor, acc.fwd)

            features[s_idx, :] = np.asarray([
                duration,
                mean_eng,
                var_ver, var_hor, var_fwd, var_eng,
                max_ver, max_hor, max_fwd, max_eng,
                min_ver, min_hor, min_fwd, min_eng,
                range_ver, range_fwd, range_eng,
                mm_ver, mm_hor, mm_fwd, mm_eng,
                sa_ver, sa_hor, sa_fwd, sa_eng,
                rms_ver, rms_hor, rms_fwd, rms_eng
            ])

        return features