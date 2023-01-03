from scipy import stats
import numpy as np
import library as sarlab


def regress_intensity(window_closure, window_amps):

    if len(window_amps.flatten()) > 2 and len(window_closure.flatten()) > 2:
        try:
            r, p = stats.pearsonr(
                window_amps.flatten(), np.angle(window_closure).flatten())

            r = r
            fitform = 'linear'

            coeff, covm = sarlab.gen_lstq(window_amps.flatten(), np.angle(window_closure).flatten(
            ), W=None, function=fitform)
        except:
            print(window_amps.flatten())
            print('')
            print(np.angle(window_closure).flatten())
            r = 0
            p = 0
            coeff = [0, 0]

        do_huber = False

        poly[j, i, :] = coeff

        rs[j, i] = r
        ps[j, i] = p

        # modeled systematic closures
        if np.abs(r) >= 0:

            est_closures = closures.eval_sytstematic_closure(
                amp_triplet_stack[:, j, i], model=coeff, form='linear')

            est_closures_int = closures.eval_sytstematic_closure(
                amp_triplet_stack[:, j, i], model=coeff, form='lineari')

            systematic_phi_errors = closures.least_norm(
                A, est_closures, pinv=False, pseudo_inv=A_dagger)

            systematic_phi_errors_int = closures.least_norm(
                A, est_closures_int, pinv=False, pseudo_inv=A_dagger)

            uncorrected_phi_errors = closures.least_norm(
                A, np.random.normal(loc=-0.02, scale=0.05, size=len(
                    closure_stack[:, j, i].flatten())), pinv=False, pseudo_inv=A_dagger)

            uncorrected_phi_errors = closures.least_norm(
                A, closure_stack[:, j, i].flatten(), pinv=False, pseudo_inv=A_dagger)

            error_coh = closures.phivec_to_coherence(
                systematic_phi_errors, coherence[:, :, j, i].shape[0])

            error_coh_int = closures.phivec_to_coherence(
                systematic_phi_errors_int, coherence[:, :, j, i].shape[0])
            error_coh_unc = closures.phivec_to_coherence(
                uncorrected_phi_errors, coherence[:, :, j, i].shape[0])

            # gradient = interpolate_phase_intensity(
            #     raw_intensities, error_coh)

            # gradient = 0
            # linear_phase = np.exp(
            #     1j * (-1 * intensities * gradient))

            # error_coh = error_coh * linear_phase

            coherence[:, :, j, i] = coherence[:,
                                              :, j, i] * error_coh.conj()

            # coherence[:, :, j, i] = error_coh

        # o
        if np.abs(r) > 1 or ((points == np.array([i, j])).all(1).any() and inputs.saveData):

            # cum_closures = np.cumsum(
            #     np.angle(window_closure).flatten()[cumulative_mask])
            # print(len(cum_closures))

            # cum_closures_slope = np.cumsum(
            #     est_closures[cumulative_mask])

            # cum_closures_int = np.cumsum(
            #     est_closures_int[cumulative_mask])

            # inten_timeseries = raw_intensities - raw_intensities[0]

            # plt.plot(cum_closures, label='Observed')
            # plt.plot(cum_closures_slope, label='Just slope')
            # plt.plot(cum_closures_int, label='Slope and Intercept')
            # plt.plot(inten_timeseries[1:-1], label='Intensities')
            # plt.legend(loc='best')
            # plt.title('Cumulative Closure Phase')
            # plt.show()
            if inputs.saveData:
                pixel_data_folder_path = f'/Users/rbiessel/Documents/InSAR/plotData/{inputs.label}/p_{i}_{j}/'

                if os.path.exists(pixel_data_folder_path):
                    print('Output folder already exists, clearing it')
                    shutil.rmtree(pixel_data_folder_path)
                os.mkdir(pixel_data_folder_path)

            print(f'point: ({i}, {j})')
            # gradient = interpolate_phase_intensity(
            #     raw_intensities, error_coh_unc, plot=True)
            # gradient = interpolate_phase_intensity(
            #     raw_intensities, error_coh, plot=True)
            print('COVARIANCE')
            if False:
                l = int(
                    np.floor(np.sqrt(ml_size[0] * ml_size[1]))/2)

                decay = np.array(
                    [0.00001, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])
                samples = 1000
                rs_decays = np.zeros((decay.shape[0], samples))
                for i in range(len(decay)):
                    C_decay = greg_sim.decay_model(
                        R=1, L=l, P=cov.cov.shape[0], coh_decay=decay[i], coh_infty=0.05, returnC=True)
                    print(C_decay.shape)
                    rs_decay = bootstrap_correlation(
                        C_decay, l, triplets, nsample=samples, fitLine=False)
                    rs_decays[i] = rs_decay

                rs_sim, coeffs_sim = bootstrap_correlation(
                    cov.cov[:, :, j, i], l, triplets, nsample=1000, fitLine=True, zeroPhi=True)

                plot_hist(rs_sim, r, rs_decays, decay)
                fig, ax = plt.subplots(ncols=3, nrows=1)
                bins = 100
                ax[0].hist(rs_sim.flatten(), bins=bins)
                ax[0].axvline(r, 0, 1, color='red')
                ax[0].set_title('Rs')

                ax[1].hist(coeffs_sim[0].flatten(), bins=bins)
                ax[1].axvline(coeff[0], 0, 1, color='red')
                ax[1].set_title('Slope')

                ax[2].hist(coeffs_sim[1].flatten(), bins=bins)
                ax[2].axvline(coeff[1], 0, 1, color='red')
                ax[2].set_title('Mean Residual Phase')

                plt.show()

            # fig, ax = plt.subplots(nrows=1, ncols=2)
            # n, bins, p = ax[0].hist(r2, bins=60)
            # ax[0].axvline(r, 0, 1, color='red')
            # ax[0].set_title('Phi = observed')
            # ax[0].set_xlabel('Correlation Coefficient')

            # n, bins, p = ax[1].hist(r2_phizero, bins=60)
            # ax[1].axvline(r, 0, 1, color='red')

            # ax[1].set_title('Phi = zero')
            # ax[1].set_xlabel('Correlation Coefficient')

            # plt.show()

            x = np.linspace(window_amps.min() - 0.1 * np.abs(window_amps.min()),
                            window_amps.max() + 0.1 * np.abs(window_amps.max()), 100)

            fig, ax = plt.subplots(figsize=(5, 2.5))

            # xy = np.vstack(
            #     [window_amps.flatten(), np.angle(window_closure).flatten()])
            # z = gaussian_kde(xy)(xy)
            ax.scatter(window_amps.flatten(), np.angle(
                window_closure).flatten(), s=10)

            ax.plot(x, closures.eval_sytstematic_closure(
                x, coeff, form=fitform), '--', label='Fit: mx')

            ax.plot(x, closures.eval_sytstematic_closure(
                x, coeff, form='lineari'), '--', label='Fit: mx+b')
            ax.axhline(y=0, color='k', alpha=0.1)
            ax.axvline(x=0, color='k', alpha=0.1)
            ax.set_xlabel('Amplitude Triplet')
            ax.set_ylabel('Closure Phase (rad)')

            handles = [mpl_patches.Rectangle((0, 0), 1, 1, fc="white", ec="white",
                                             lw=0, alpha=0)] * 2

            # create the corresponding number of labels (= the text you want to display)
            labels = []
            labels.append(f'R$^{{2}} = {{{np.round(r**2, 2)}}}$')

            # create the legend, supressing the blank space of the empty line symbol and the
            # padding between symbol and label by setting handlelenght and handletextpad
            ax.legend(handles, labels, loc='best', fontsize='large',
                      fancybox=True, framealpha=0.7,
                      handlelength=0, handletextpad=0)

            plt.tight_layout()
            plt.savefig(os.path.join(
                pixel_data_folder_path, 'scatter.png'), dpi=200)
            np.save(os.path.join(pixel_data_folder_path,
                                 'ampTriplets.np'), window_amps.flatten())
            np.save(os.path.join(pixel_data_folder_path, 'closures.np'), np.angle(
                window_closure).flatten())
            np.save(os.path.join(
                pixel_data_folder_path, 'coeff.np'), coeff)
            np.save(os.path.join(
                pixel_data_folder_path, 'C_raw.np'), uncorrected[:, :, j, i])
            np.save(os.path.join(
                pixel_data_folder_path, 'C_ln_slope.np'), error_coh)
            np.save(os.path.join(
                pixel_data_folder_path, 'C_ln_unc.np'), error_coh_unc)
            np.save(os.path.join(
                pixel_data_folder_path, 'Intensities.np'), raw_intensities)

            # plt.show()

            fig, ax = plt.subplots(
                nrows=3, ncols=1, figsize=(5, 5))
            # slope_stderr = np.sqrt(covm[0][0])
            # intercept_stderr = np.sqrt(covm[1][1])

            xy = np.vstack(
                [window_amps.flatten(), np.angle(window_closure).flatten()])
            z = gaussian_kde(xy)(xy)

            ax[0].scatter(window_amps.flatten(), np.angle(
                window_closure).flatten(), c=z, s=10)  # alpha=(alphas)**(1))

            ax[0].plot(x, closures.eval_sytstematic_closure(
                x, coeff, form=fitform), '--', label='Fit: mx')

            ax[0].plot(x, closures.eval_sytstematic_closure(
                x, coeff, form='lineari'), '--', label='Fit: mx+b')
            ax[0].set_title(f'r: {np.round(r, 3)}')
            ax[0].set_ylabel(r'$\Xi$')
            ax[0].set_xlabel(
                'Intensity Triplet')
            ax[0].legend(bbox_to_anchor=(1.05, 0.5),
                         loc='center left', borderaxespad=0.)

            ax[1].set_xlabel('Intensity Ratio (dB)')
            ax[1].set_ylabel('Estimated Phase Error (rad)')

            iratios = closures.coherence_to_phivec(intensities)

            nl_phases_uncorrected = np.angle(
                closures.coherence_to_phivec(error_coh_unc))

            nlphases = np.angle(
                closures.coherence_to_phivec(error_coh))

            nlphases_int = np.angle(
                closures.coherence_to_phivec(error_coh_int))

            # print(m)
            x = np.linspace(iratios.min(
            ) - 0.1 * np.abs(iratios.min()), iratios.max() + 0.1 * np.abs(iratios.max()), 100)

            max_range = np.max(np.abs(iratios))  # + 0.1
            x = np.linspace(-max_range, max_range, 100)

            # ax[1].scatter(iratios, nlphases, s=15,
            #               marker='x', color='blue', label='/w a linear component to force monotonicity')

            # ax[1].scatter(iratios, nl_phases_uncorrected, s=10,
            #               marker='x', color='black', label='From Uncorrected Closures')

            # ax[1].scatter(iratios, nlphases_int, s=15,
            #               marker='x', color='orange', label='With Intercept')

            ax[1].scatter(iratios, nlphases, s=20,
                          marker='x', color='blue', label='mx')
            ax[1].axhline(y=0, color='k', alpha=0.1)
            ax[1].axvline(x=0, color='k', alpha=0.1)

            ax[1].legend(bbox_to_anchor=(1.05, 0.5),
                         loc='center left', borderaxespad=0.)

            residual_closure = np.angle(window_closure).flatten() - closures.eval_sytstematic_closure(
                window_amps.flatten(), coeff, form='linear')

            indexsort = np.argsort(baslines_b)
            residual_closure = residual_closure[indexsort]
            baslines_b = baslines_b[indexsort]

            u, s = np.unique(baslines_b, return_index=True)
            split_residuals_b = np.split(residual_closure, s[1:])

            u, s = np.unique(baslines_b, return_index=True)
            split_residuals_b = np.split(residual_closure, s[1:])

            # ax[2].scatter(
            #     baslines_a, residual_closure, s=10, label='a')
            ax[2].boxplot(split_residuals_b,
                          positions=u, widths=9)
            # ax[2].boxplot(split_residuals_b)

            # ax[2].scatter(
            #     baslines_b, residual_closure, s=10, label='b', alpha=0.5)
            ax[2].set_xlabel('basline')
            ax[2].set_ylabel('Closures')
            ax[2].legend(loc='lower right')

            plt.tight_layout()
            plt.show()

            fig, ax = plt.subplots(ncols=2, nrows=1)
            ax[0].hist(np.angle(window_closure).flatten(), bins=50)
            # ax[0].set_title()
            ax[1].hist(window_amps.flatten(), bins=50)
            ax[1].set_title('Amp Triplet')

            plt.show()

            fig, ax = plt.subplots(nrows=1, ncols=2)

            residual = np.angle(window_closure.flatten() *
                                np.exp(1j * -1 * est_closures))
            ax[0].hist(
                np.angle(np.exp(1j*est_closures)).flatten(), bins=60, density=True)
            ax[0].set_title('Predicted')

            ax[1].hist(residual, bins=60, density=True)
            ax[1].set_title(
                f'Residual Mean: {np.round(np.mean(residual), 2)} -- Intercept: {np.round(coeff[1], 2)}')
            plt.show()

            fig, ax = plt.subplots(nrows=1, ncols=3)
            ax[0].set_title(
                'Estimated Phase Error -- slope')

            ax[0].imshow(np.angle(error_coh),
                         vmin=-np.pi/15, vmax=np.pi/15, cmap=plt.cm.seismic)

            ax[0].set_xlabel('Reference Image')
            ax[0].set_ylabel('Secondary Image')
            ax[1].set_title(
                'Estimated Phase Error -- intercept')
            im = ax[1].imshow(np.angle(error_coh_int * error_coh.conj()),
                              vmin=-np.pi/15, vmax=np.pi/15, cmap=plt.cm.seismic)

            ax[1].set_xlabel('Reference Image')
            ax[1].set_ylabel('Secondary Image')

            ax[2].set_title(
                'Estimated Phase Error -- all')
            im = ax[2].imshow(
                np.angle(error_coh_unc), vmin=-np.pi/5, vmax=np.pi/5, cmap=plt.cm.seismic)

            ax[2].set_xlabel('Reference Image')
            ax[2].set_ylabel('Secondary Image')

            fig.subplots_adjust(right=0.8)
            cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
            fig.colorbar(im, cax=cbar_ax,
                         label='Estimated Nonlinear Phase Error (rad)')
            plt.show()

        else:
            r = 0
        # except:
        #     # print('robust regression failed :(')
        #     poly[j, i, :] = np.zeros((2))
        #     rs[j, i] = 0
