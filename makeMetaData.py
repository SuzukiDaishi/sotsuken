import util

# 音素全体の音素を取得する
# FIXME: 「[1]    4610 segmentation fault  python makeMetaData.py」がたまに出るので

def main():
    #makePicle('./datasets/suzuki', './cache36_suzuki.pkl')
    makePicle('./datasets/kinoshita/overdrive', './cache36_kinoshita.pkl')


def makePicle(url, savefile):
    waves = util.loadWaves(url)
    f0s, sps, aps, coded_sps = util.worldEncodeData(waves)
    log_f0s_mean, log_f0s_std = util.logf0Statistics(f0s)
    coded_sps_transposed = util.transposeInList(coded_sps)
    coded_sps_norm, coded_sps_mean, coded_sps_std, coded_sps_max = util.codedSpsNormalizationFitTransoform_fix(
        coded_sps_transposed, use_max=True)
    data = coded_sps_norm, coded_sps_mean, coded_sps_std, coded_sps_max, log_f0s_mean, log_f0s_std
    util.savePickle(savefile, data)
    print('Preprocessing Done.')

if __name__ == "__main__":
    main()
