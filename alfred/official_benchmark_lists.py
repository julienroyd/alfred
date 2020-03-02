searches_list = [
    # # DDPG_benchmarkv1 -----
    "Ju59_fa3e6b1_03c955b_DDPG_spread_random_benchmarkv1",
    "Ju60_fa3e6b1_03c955b_DDPG_bounce_random_benchmarkv1",
    "PB59_8e67439_aa00755_DDPG_compromise_random_mean_perf_benchmarkv1",
    "Ju62_fa3e6b1_03c955b_DDPG_chase_random_benchmarkv1",

    # MADDPG_benchmarkv1 -----
    "PB7_fa3e6b1_03c955b_MADDPG_spread_random_benchmarkv1",
    "PB8_fa3e6b1_03c955b_MADDPG_bounce_random_benchmarkv1",
    "PB58_8e67439_aa00755_MADDPG_compromise_random_mean_perf_benchmarkv1",
    "PB10_fa3e6b1_03c955b_MADDPG_chase_random_benchmarkv1",

    # # SharedMADDPG_benchmarkv1 -----
    "PB33_c3eb925_861f734_SharedMADDPG_spread_fix_benchmarkv1",
    "PB34_c3eb925_861f734_SharedMADDPG_bounce_fix_benchmarkv1",
    "Ju83_050ca34_aa00755_SharedMADDPG_compromise_random_meanReturn_benchmarkv1",
    "PB32_c3eb925_861f734_SharedMADDPG_chase_fix_benchmarkv1",

    # # TeamMADDPG ablation
    # "Ju74_2c09740_861f734_TeamMADDPGablation_spread_random_benchmarkv1_ablationLamda2off",
    # "Ju76_2c09740_861f734_TeamMADDPGablation_bounce_random_benchmarkv1_ablationLamda2off",
    # "Ju84_bf550ec_aa00755_TeamMADDPGablation_compromise_random_meanReturn_benchmarkv1_ablationLamda2off",
    # "Ju75_2c09740_861f734_TeamMADDPGablation_chase_random_benchmarkv1_ablationLamda2off",

    # # CoachMADDPG ablation
    # "PB50_47f2300_3864976_CoachMADDPGablation_spread_review",
    # "PB51_5722b30_3864976_CoachMADDPGablation_bounce_review",
    # "PB61_9334291_aa00755_CoachMADDPGablation_compromise_random_review_mean_perf_benchmarkv1",
    # "PB52_5722b30_3864976_CoachMADDPGablation_chase_review",

    # TeamMADDPG_benchmarkv1 -----
    "Ju53_1756c4b_03c955b_TeamMADDPG_spread_random_benchmarkv1",
    "Ju54_1756c4b_03c955b_TeamMADDPG_bounce_random_benchmarkv1",
    "Ju85_bf550ec_aa00755_TeamMADDPG_compromise_random_meanReturn_benchmarkv1",
    "Ju56_1756c4b_03c955b_TeamMADDPG_chase_random_benchmarkv1",

    # CoachMADDPG_benchmarkv1 -----
    "PB1_2bc3c27_5f7a15b_CoachMADDPG_spread_random_review",
    "PB2_2bc3c27_5f7a15b_CoachMADDPG_bounce_random_review",
    "PB60_9334291_aa00755_CoachMADDPG_compromise_random_review_mean_perf_benchmarkv1",
    "PB3_2bc3c27_5f7a15b_CoachMADDPG_chase_random_review",

]

retrains_list = [
    # DDPG_benchmarkv1 -----
    "Ju66_fa3e6b1_03c955b_DDPG_spread_retrainBestJu59_benchmarkv1",
    "Ju67_fa3e6b1_03c955b_DDPG_bounce_retrainBestJu60_benchmarkv1",
    "PB60_8e67439_aa00755_DDPG_compromise_retrainBestPB59_mean_perf_benchmarkv1",
    "Ju65_fa3e6b1_03c955b_DDPG_chase_retrainBestJu62_benchmarkv1",

    # MADDPG_benchmarkv1 -----
    "PB14_fa3e6b1_03c955b_MADDPG_spread_retrainBestPB7_benchmarkv1",
    "PB13_fa3e6b1_03c955b_MADDPG_bounce_retrainBestPB8_benchmarkv1",
    "PB61_8e67439_aa00755_MADDPG_compromise_retrainBestPB58_mean_perf_benchmarkv1",
    "PB11_60af96a_03c955b_MADDPG_chase_retrainBestPB10_benchmarkv1",

    # # SharedMADDPG_benchmarkv1 -----
    "PB37_22c465a_861f734_SharedMADDPG_spread_fix_benchmarkv1_retrainBestPB33",
    "PB38_22c465a_861f734_SharedMADDPG_bounce_fix_benchmarkv1_retrainBestPB34",
    "Ju86_050ca34_aa00755_SharedMADDPG_compromise_retrainBestJu83_meanReturn_benchmarkv1",
    "PB36_22c465a_861f734_SharedMADDPG_chase_fix_benchmarkv1_retrainBestPB32",

    # # TeamMADDPG ablation
    # "Ju78_2c09740_861f734_TeamMADDPGablation_spread_retrainBestJu74_benchmarkv1_ablationLamda2off",
    # "Ju80_2c09740_861f734_TeamMADDPGablation_bounce_retrainBestJu76_benchmarkv1_ablationLamda2off",
    # "Ju88_bf550ec_aa00755_TeamMADDPGablation_compromise_retrainBestJu84_meanReturn_benchmarkv1_ablationLamda2off",
    # "Ju79_2c09740_861f734_TeamMADDPGablation_chase_retrainBestJu75_benchmarkv1_ablationLamda2off",

    # # CoachMADDPG ablation
    # "PB57_5722b30_3864976_CoachMADDPGablation_spread_review_retrainBestPB50",
    # "PB54_5722b30_3864976_CoachMADDPGablation_bounce_review_retrainBestPB51",
    # "PB63_9334291_aa00755_CoachMADDPGablation_compromise_retrainBestPB61_review_mean_perf_benchmarkv1",
    # "PB55_5722b30_3864976_CoachMADDPGablation_chase_review_retrainBestPB52",

    # TeamMADDPG_benchmarkv1 -----
    "PB15_6afa025_5f7a15b_TeamMADDPG_spread_retrainBestJu53_benchmarkv1",
    "PB16_6afa025_5f7a15b_TeamMADDPG_bounce_retrainBestJu54_benchmarkv1",
    "Ju87_bf550ec_aa00755_TeamMADDPG_compromise_retrainBestJu85_meanReturn_benchmarkv1",
    "PB17_6afa025_5f7a15b_TeamMADDPG_chase_retrainBestJu56_benchmarkv1",

    # CoachMADDPG benchmarkv1 -----
    "PB4_2bc3c27_5f7a15b_CoachMADDPG_spread_retrainBestPB1_review",
    "PB5_2bc3c27_5f7a15b_CoachMADDPG_bounce_retrainBestPB2_review",
    "PB62_9334291_aa00755_CoachMADDPG_compromise_retrainBestPB60_review_mean_perf_benchmarkv1",
    "PB6_2bc3c27_5f7a15b_CoachMADDPG_chase_retrainBestPB3_review",

]
