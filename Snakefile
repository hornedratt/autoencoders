NOISES = [10, 20, 30, 40, 50]

rule all:
    input:
#       наборы данных
        "data\\processed\\original_MS_profiles.csv",
        # "data\\processed\\original_MS_profiles.pkl",
        [f"data\\processed\\sets\\set_normal_noise_{noise}%.pkl" for noise in NOISES],
        [f"data\\processed\\sets\\test_normal_noise_{noise}%.pkl" for noise in NOISES],

        # "reports\\figures\\tSNE_orig.png",
        # "reports\\Hopkins_orig.txt",
        # [f"reports\\figures\\tSNE_{noise}%.png" for noise in NOISES],
        # [f"reports\\Hopkins_{noise}%.txt" for noise in NOISES],

#       отчеты о тренировке кодера
        [f"reports\\figures\\DAE_norm_noise_{noise}%.png" for noise in NOISES],
        [f"reports\\figures\\heat_map_group_{noise}%.png" for noise in NOISES],
        [f"reports\\figures\\heat_map_ID_{noise}%.png" for noise in NOISES],

#       модели: кодер и два леса
        [f"models\\DAE_norm_noise_{noise}%.pkl" for noise in NOISES],
        [f"models\\forest_{noise}%_group" for noise in NOISES],
        [f"models\\forest_{noise}%_ID" for noise in NOISES],

#       отчеты о тренировке лесов
        [f"reports\\forest_{noise}%_group.csv" for noise in NOISES],
        [f"reports\\forest_{noise}%_ID.csv" for noise in NOISES],
        [f"reports\\figures\\forest_{noise}%_importances_group.png" for noise in NOISES],
        [f"reports\\figures\\forest_{noise}%_importances_ID.png" for noise in NOISES],
        [f"reports\\mz_features_{noise}%_group.txt" for noise in NOISES],
        [f"reports\\mz_features_{noise}%_ID.txt" for noise in NOISES],
        f"reports\\figures\\cross_valid_40%_result_group.png",

#       отчеты о использовании разных наборов данных с моделями, обученными на
#       разных наборах
        "reports\\cross_noise_acc_group.csv",
        "reports\\cross_noise_acc_ID.csv",
        "reports\\cross_noise_f1_group.csv",
        "reports\\cross_noise_f1_ID.csv"

gpu_group = 'gpu_tasks'

rule make_original_profiles_csv:
    input:
        "data\\raw"
    output:
        "data\\processed\\original_MS_profiles.csv"
    shell:
        "python -m src.data.get_csv_with_original_profiles {input} {output}"

rule wrap_orig_data:
    shell:
        "python -m src.data.wraper_for_orig_data"

rule add_normal_noise:
    input:
        "data\\processed\\original_MS_profiles.csv"
    output:
        "data\\processed\\sets\\set_normal_noise_{noise}%.pkl"
    wildcard_constraints:
        noise="(10|20|30|40|50)"
    shell:
        "python -m src.data.test_noise {input} {output} --noise {wildcards.noise}"

# rule preanalysis_orig:
#     input:
#         "data\\processed\\original_MS_profiles.pkl"
#     output:
#         "reports\\figures\\tSNE_orig.png",
#         "reports\\Hopkins_orig.txt"
#     shell:
#         "python -m src.features.preanalysis {input} {output}"

# rule preanalysis_noise:
#     input:
#         "data\\processed\\sets\\set_normal_noise_{noise}%.pkl"
#     output:
#         "reports\\figures\\tSNE_{noise}%.png",
#         "reports\\Hopkins_{noise}%.txt"
#
#     wildcard_constraints:
#         noise="(10|20|30|40|50)"
#     shell:
#         "python -m src.features.preanalysis {input} {output}"

rule train_autoencoder:
    output:
        "models\\DAE_norm_noise_{noise}%.pkl",
        "reports\\figures\\DAE_norm_noise_{noise}%.png"
    wildcard_constraints:
        noise="(10|20|30|40|50)"
    group:
        gpu_group
    resources:
        gpu=1
    shell:
        "python -m src.models.train {output} --seed {wildcards.noise} --noise_factor {wildcards.noise}"

rule heat_map:
    input:
        "models\\DAE_norm_noise_{noise}%.pkl",
        "data\\processed\\sets\\set_normal_noise_{noise}%.pkl"
    output:
        "reports\\figures\\heat_map_group_{noise}%.png",
        "reports\\figures\\heat_map_ID_{noise}%.png"
    wildcard_constraints:
        noise="(10|20|30|40|50)"
    shell:
        "python -m src.visualization.heat_map {input} {output}"

rule train_forest:
    input:
        "models\\DAE_norm_noise_{noise}%.pkl",
        "data\\processed\\sets\\set_normal_noise_{noise}%.pkl"
    output:
        "data\\processed\\sets\\test_normal_noise_{noise}%.pkl",
        "models\\forest_{noise}%_group",
        "reports\\forest_{noise}%_group.csv",
        "models\\forest_{noise}%_ID",
        "reports\\forest_{noise}%_ID.csv"
    wildcard_constraints:
        noise="(10|20|30|40|50)"
    shell:
        "python -m src.models.train_forest {input} {output} --seed {wildcards.noise}"
rule cross_validation:
    input:
        "data\\processed\\sets\\set_normal_noise_{noise}%.pkl",
        "models\\DAE_norm_noise_{noise}%.pkl"
    output:
        "reports\\figures\\cross_valid_{noise}%_result_group.png",
    wildcard_constraints:
        # noise="(10|20|30|40|50)"
        noise="(40)"
    shell:
        "python -m src.models.cross_valid {input} {output} --amount 1000"

rule importance_analysis:
    input:
        "data\\processed\\sets\\test_normal_noise_{noise}%.pkl",
        "models\\DAE_norm_noise_{noise}%.pkl",
        "models\\forest_{noise}%_group",
        "models\\forest_{noise}%_ID"
    output:
        "reports\\figures\\forest_{noise}%_importances_group.png",
        "reports\\figures\\forest_{noise}%_importances_ID.png",
        "reports\\mz_features_{noise}%_group.txt",
        "reports\\mz_features_{noise}%_ID.txt"
    wildcard_constraints:
        noise="(10|20|30|40|50)"
    shell:
        "python -m src.features.importance_analysis {input} {output}"
rule cross_noise:
    input:
        [f"data\\processed\\sets\\test_normal_noise_{noise}%.pkl" for noise in NOISES],
        [f"models\\DAE_norm_noise_{noise}%.pkl" for noise in NOISES],
        [f"models\\forest_{noise}%_group" for noise in NOISES],
        [f"models\\forest_{noise}%_ID" for noise in NOISES]
    output:
        "reports\\cross_noise_acc_group.csv",
        "reports\\cross_noise_acc_ID.csv",
        "reports\\cross_noise_f1_group.csv",
        "reports\\cross_noise_f1_ID.csv"
    shell:
        "python -m src.models.cross_noise {{output}} {noises}".format(noises=",".join(map(str, NOISES)))
