import sys
assert sys.version_info >= (3, 5) # make sure we have Python 3.5+

import pandas as pd


# ------------------------Function to combine df of all gear type-----------------------------
def main(p1, p2, p3, p4, p5, p6, version):
    df1 = pd.read_csv('../data/' + p1)
    df2 = pd.read_csv('../data/' + p2)
    df3 = pd.read_csv('../data/' + p3)
    df4 = pd.read_csv('../data/' + p4)
    df5 = pd.read_csv('../data/' + p5)
    df6 = pd.read_csv('../data/' + p6)

    df_all = pd.concat([df1, df2, df3, df4, df5, df6])
    print(df_all.shape)
    df_all.to_csv('../data/combine_gear_' + version + '.csv', index=False)


if __name__ == '__main__':
    p1 = sys.argv[1]
    p2 = sys.argv[2]
    p3 = sys.argv[3]
    p4 = sys.argv[4]
    p5 = sys.argv[5]
    p6 = sys.argv[6]
    version = sys.argv[7]
    main(p1, p2, p3, p4, p5, p6, version)
