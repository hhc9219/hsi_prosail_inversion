def set_globals():
    global THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER
    THREADS, MEMORY, HSI_CONFIG, PROJECT_FOLDER, OUTPUT_FOLDER = get_persistent_config_data(__file__)


def main():
    import numpy as np
    from matplotlib import pyplot as plt

    HSI_NAME = input("Please enter the hsi to process from hsi_config.json: ")
    inv_path = OUTPUT_FOLDER / f"{HSI_NAME}_inv_res.npy"

    CMAP = "viridis"

    FIGURES_FOLDER = OUTPUT_FOLDER / "figures"
    FIGURES_FOLDER.mkdir(parents=True, exist_ok=True)

    inverted_prosail = np.load(inv_path)

    success = inverted_prosail[:, :, 0]
    n = inverted_prosail[:, :, 1]
    cab = inverted_prosail[:, :, 2]
    ccx = inverted_prosail[:, :, 3]
    ewt = inverted_prosail[:, :, 4]
    lma = inverted_prosail[:, :, 5]
    lai = inverted_prosail[:, :, 6]
    psoil = inverted_prosail[:, :, 7]
    mask = inverted_prosail[:, :, 8]

    where_inverted = success > 0.99
    percent_successful = success.round().sum() / mask.round().sum()

    n_mean = np.mean(n[where_inverted])
    cab_mean = np.mean(cab[where_inverted])
    ccx_mean = np.mean(ccx[where_inverted])
    ewt_mean = np.mean(ewt[where_inverted])
    lma_mean = np.mean(lma[where_inverted])
    lai_mean = np.mean(lai[where_inverted])
    psoil_mean = np.mean(psoil[where_inverted])

    n[~where_inverted] = n_mean
    cab[~where_inverted] = cab_mean
    ccx[~where_inverted] = ccx_mean
    ewt[~where_inverted] = ewt_mean
    lma[~where_inverted] = lma_mean
    lai[~where_inverted] = lai_mean
    psoil[~where_inverted] = psoil_mean

    param_names = ["Success", "N", "CAB", "CCX", "EWT", "LMA", "LAI", "PSOIL", "Mask"]
    inv_params = [success, n, cab, ccx, ewt, lma, lai, psoil, mask]

    for name, inv_param in zip(param_names, inv_params):
        plt.figure()
        plt.imshow(inv_param, cmap=CMAP, interpolation="none")
        plt.colorbar()
        title = name if name != "Success" else name + ": " + str(int(round(percent_successful * 100))) + "%"
        plt.title(title)
        plt.savefig(FIGURES_FOLDER / (name + ".png"))
        plt.close()

    print("Saved result to output folder.")


if __name__ == "__main__":
    from modules.environment_manager import enforce_venv, get_persistent_config_data

    enforce_venv(__file__)
    set_globals()
    main()
