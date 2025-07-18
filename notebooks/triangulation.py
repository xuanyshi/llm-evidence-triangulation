import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

__all__ = ['analyze_tri_df', 'plot_cumulative_trends']

def analyze_tri_df(input_df,detail_info):
    # Separate MR from non-MR
    df_non_mr = input_df[input_df['study_design'] != 'MR']
    df_mr     = input_df[input_df['study_design'] == 'MR']

    df_non_mr = df_non_mr[df_non_mr['exposure_direction'].isin(['increased','decreased'])]
    df_non_mr = df_non_mr[df_non_mr['direction'].isin(['increase','decrease','no_change'])]
    # Calculate 5th and 95th percentiles for the non-MR group
    lower_bound = df_non_mr['number_of_participants'].quantile(0.05)
    upper_bound = df_non_mr['number_of_participants'].quantile(0.95)

    # Filter the non-MR group to keep only rows within these quantiles
    df_non_mr_filtered = df_non_mr[
        (df_non_mr['number_of_participants'] >= lower_bound) &
        (df_non_mr['number_of_participants'] <= upper_bound)
    ]

    # Combine MR (unfiltered) and filtered Non-MR
    input_df = pd.concat([df_non_mr_filtered, df_mr], ignore_index=True)

    tri_df = (
        input_df
        .groupby(['study_design', 'exposure_direction', 'direction'], as_index=False)
        .agg(
            count=('direction', 'size'),  # equivalent to .count(), but simpler for counting rows
            participants_sum=('number_of_participants', 'sum')
        )
    )

    print(input_df.shape,input_df['pmid'].nunique())
    # Group by and reset index as in the provided code
    if detail_info==True:
      print(tri_df)
    else:
      pass

    # Define a helper function to handle zeros and summing
    def handle_zero_and_sum(*args):
        #print(args)
        adjusted_values = [val + 0.1 if val == 0 else val for val in args]
        #print(adjusted_values)
        total_sum = sum(adjusted_values)
        return adjusted_values, total_sum

    # Extract and print values for RCT with different exp_direction and direction
    rct_i_i = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    rct_i_i_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'participants_sum'].sum()

    rct_i_n = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    rct_i_n_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'participants_sum'].sum()

    rct_i_d = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    rct_i_d_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'participants_sum'].sum()

    denominator_rct_i = rct_i_i_ppl + rct_i_n_ppl + rct_i_d_ppl
    rct_i_i = rct_i_i * (rct_i_i_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
    print(f"Ratio (rct_i_i_ppl / total): {rct_i_i_ppl / denominator_rct_i:.2f}" if denominator_rct_i != 0 else "Ratio (rct_i_i_ppl / total): 0.00")
    rct_i_n = rct_i_n * (rct_i_n_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
    print(f"Ratio (rct_i_n_ppl / total): {rct_i_n_ppl / denominator_rct_i:.2f}" if denominator_rct_i != 0 else "Ratio (rct_i_n_ppl / total): 0.00")
    rct_i_d = rct_i_d * (rct_i_d_ppl / denominator_rct_i) if denominator_rct_i != 0 else 0
    print(f"Ratio (rct_i_d_ppl / total): {rct_i_d_ppl / denominator_rct_i:.2f}" if denominator_rct_i != 0 else "Ratio (rct_i_d_ppl / total): 0.00")

    (rct_i_i, rct_i_n, rct_i_d), rct_i_sum = handle_zero_and_sum(rct_i_i, rct_i_n, rct_i_d)
    # print(f"RCT Increased Increase: {rct_i_i}, RCT Increased Not Found: {rct_i_n}, RCT Increased Decrease: {rct_i_d}")
    # print(f"Sum for RCT Increased: {rct_i_sum}")

    rct_d_i = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    rct_d_i_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'participants_sum'].sum()

    rct_d_n = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    rct_d_n_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'participants_sum'].sum()

    rct_d_d = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    rct_d_d_ppl = tri_df.loc[(tri_df['study_design'] == 'RCT') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'participants_sum'].sum()

    denominator_rct_d = rct_d_i_ppl + rct_d_n_ppl + rct_d_d_ppl
    rct_d_i = rct_d_i * (rct_d_i_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
    print(f"Ratio (rct_d_i_ppl / total): {rct_d_i_ppl / denominator_rct_d:.2f}" if denominator_rct_d != 0 else "Ratio (rct_d_i_ppl / total): 0.00")
    rct_d_n = rct_d_n * (rct_d_n_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
    print(f"Ratio (rct_d_n_ppl / total): {rct_d_n_ppl / denominator_rct_d:.2f}" if denominator_rct_d != 0 else "Ratio (rct_d_n_ppl / total): 0.00")
    rct_d_d = rct_d_d * (rct_d_d_ppl / denominator_rct_d) if denominator_rct_d != 0 else 0
    print(f"Ratio (rct_d_d_ppl / total): {rct_d_d_ppl / denominator_rct_d:.2f}" if denominator_rct_d != 0 else "Ratio (rct_d_d_ppl / total): 0.00")

    (rct_d_i, rct_d_n, rct_d_d), rct_d_sum = handle_zero_and_sum(rct_d_i, rct_d_n, rct_d_d)
    # print(f"RCT Decreased Increase: {rct_d_i}, RCT Decreased Not Found: {rct_d_n}, RCT Decreased Decrease: {rct_d_d}")
    # print(f"Sum for RCT Decreased: {rct_d_sum}")

    # Extract and print values for MR with different exp_direction and directionb
    mr_i_i = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    mr_i_n = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    mr_i_d = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    (mr_i_i, mr_i_n, mr_i_d), mr_i_sum = handle_zero_and_sum(mr_i_i, mr_i_n, mr_i_d)
    # print(f"MR Increased Increase: {mr_i_i}, MR Increased Not Found: {mr_i_n}, MR Increased Decrease: {mr_i_d}")
    # print(f"Sum for MR Increased: {mr_i_sum}")

    mr_d_i = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    mr_d_n = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    mr_d_d = tri_df.loc[(tri_df['study_design'] == 'MR') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    (mr_d_i, mr_d_n, mr_d_d), mr_d_sum = handle_zero_and_sum(mr_d_i, mr_d_n, mr_d_d)
    # print(f"MR Decreased Increase: {mr_d_i}, MR Decreased Not Found: {mr_d_n}, MR Decreased Decrease: {mr_d_d}")
    # print(f"Sum for MR Decreased: {mr_d_sum}")

    # Extract and print values for OS with different exp_direction and direction
    os_i_i = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    os_i_i_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'increase'), 'participants_sum'].sum()

    os_i_n = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    os_i_n_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'no_change'), 'participants_sum'].sum()

    os_i_d = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    os_i_d_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'increased') & (tri_df['direction'] == 'decrease'), 'participants_sum'].sum()

    denominator_os_i = os_i_i_ppl + os_i_n_ppl + os_i_d_ppl
    os_i_i = os_i_i * (os_i_i_ppl / denominator_os_i) if denominator_os_i != 0 else 0
    print(f"Ratio (os_i_i_ppl / total): {os_i_i_ppl / denominator_os_i:.2f}" if denominator_os_i != 0 else "Ratio (os_i_i_ppl / total): 0.00")
    os_i_n = os_i_n * (os_i_n_ppl / denominator_os_i) if denominator_os_i != 0 else 0
    print(f"Ratio (os_i_n_ppl / total): {os_i_n_ppl / denominator_os_i:.2f}" if denominator_os_i != 0 else "Ratio (os_i_n_ppl / total): 0.00")
    os_i_d = os_i_d * (os_i_d_ppl / denominator_os_i) if denominator_os_i != 0 else 0
    print(f"Ratio (os_i_d_ppl / total): {os_i_d_ppl / denominator_os_i:.2f}" if denominator_os_i != 0 else "Ratio (os_i_d_ppl / total): 0.00")

    (os_i_i, os_i_n, os_i_d), os_i_sum = handle_zero_and_sum(os_i_i, os_i_n, os_i_d)
    # print(f"OS Increased Increase: {os_i_i}, OS Increased Not Found: {os_i_n}, OS Increased Decrease: {os_i_d}")
    # print(f"Sum for OS Increased: {os_i_sum}")

    os_d_i = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'count'].sum()
    os_d_i_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'increase'), 'participants_sum'].sum()

    os_d_n = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'count'].sum()
    os_d_n_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'no_change'), 'participants_sum'].sum()

    os_d_d = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'count'].sum()
    os_d_d_ppl = tri_df.loc[(tri_df['study_design'] == 'OS') & (tri_df['exposure_direction'] == 'decreased') & (tri_df['direction'] == 'decrease'), 'participants_sum'].sum()

    denominator_os_d = os_d_i_ppl + os_d_n_ppl + os_d_d_ppl
    os_d_i = os_d_i * (os_d_i_ppl / denominator_os_d) if denominator_os_d != 0 else 0
    print(f"Ratio (os_d_i_ppl / total): {os_d_i_ppl / denominator_os_d:.2f}" if denominator_os_d != 0 else "Ratio (os_d_i_ppl / total): 0.00")
    os_d_n = os_d_n * (os_d_n_ppl / denominator_os_d) if denominator_os_d != 0 else 0
    print(f"Ratio (os_d_n_ppl / total): {os_d_n_ppl / denominator_os_d:.2f}" if denominator_os_d != 0 else "Ratio (os_d_n_ppl / total): 0.00")
    os_d_d = os_d_d * (os_d_d_ppl / denominator_os_d) if denominator_os_d != 0 else 0
    print(f"Ratio (os_d_d_ppl / total): {os_d_d_ppl / denominator_os_d:.2f}" if denominator_os_d != 0 else "Ratio (os_d_d_ppl / total): 0.00")

    (os_d_i, os_d_n, os_d_d), os_d_sum = handle_zero_and_sum(os_d_i, os_d_n, os_d_d)

    # Calculate probabilities
    p_excitatory = 1/6 * ((rct_i_i/rct_i_sum) + (mr_i_i/mr_i_sum) + (os_i_i/os_i_sum) + (rct_d_d/rct_d_sum) + (mr_d_d/mr_d_sum) + (os_d_d/os_d_sum))
    #print(rct_i_i,rct_i_sum)
    p_no_change = 1/6 * ((rct_i_n/rct_i_sum) + (mr_i_n/mr_i_sum) + (os_i_n/os_i_sum) + (rct_d_n/rct_d_sum) + (mr_d_n/mr_d_sum) + (os_d_n/os_d_sum))
    p_inhibitory = 1/6 * ((rct_i_d/rct_i_sum) + (mr_i_d/mr_i_sum) + (os_i_d/os_i_sum) + (rct_d_i/rct_d_sum) + (mr_d_i/mr_d_sum) + (os_d_i/os_d_sum))

    # Calculate LOE
    biggest = max(p_excitatory, p_no_change, p_inhibitory)
    loe = (biggest - (1/3)) / (1 - (1/3))

    # Determine which relationship has the biggest probability
    if biggest == p_excitatory:
        biggest_relation = 'excitatory'
    elif biggest == p_no_change:
        biggest_relation = 'no_change'
    else:
        biggest_relation = 'inhibitory'
    # Round probabilities and LOE
    return {
        "p_excitatory": round(p_excitatory, 3),
        "p_no_change": round(p_no_change, 3),
        "p_inhibitory": round(p_inhibitory, 3),
        "loe": round(loe, 3),
        "biggest": biggest_relation
    }

def plot_cumulative_trends(score_df_yearly_dfs,
                           start_display=1980,
                           end_display=2020,
                           save_path=None,
                           focus_year=None,
                           source_data_path=None,
                           base_fontsize=16):
    """
    Plots cumulative trends for p_excitatory, p_no_change, p_inhibitory,
    and cumulative evidence counts (no rolling average).

    Args
    ----
    score_df_yearly_dfs : list[pd.DataFrame]
        One DataFrame per publication year.
    start_display : int
        First year shown on the x-axis.
    end_display : int
        Last year shown on the x-axis.
    save_path : str | None
        Optional path to save the PNG (dpi=600).
    focus_year : int | None
        If provided, draw a vertical dashed line at this year.
    source_data_path : str | None
        If provided, write the subsetted results to CSV.
    base_fontsize : int
        Base font size for every text element.
    """

    # ──────────────────────────────── style ────────────────────────────
    plt.rcParams.update({'font.size': base_fontsize,
                         'font.family':'sans-serif',
                         'font.sans-serif':['DejaVu Sans']})

    # ──────────────────────── accumulate yearly stats ──────────────────
    results = []
    cumulative_counts = dict(p_excitatory=0, p_no_change=0, p_inhibitory=0)

    all_years = sorted({int(y[0]) for df in score_df_yearly_dfs
                        for y in [df['pub_year'].unique()] if len(y) > 0})

    prev_loc, prev_biggest = None, None

    for end_year in all_years:
        cumulative_dfs = [df for df in score_df_yearly_dfs
                          if (not df.empty
                              and int(df['pub_year'].unique()[0]) <= end_year)]

        if not cumulative_dfs:
            continue

        combined_df   = pd.concat(cumulative_dfs)
        current_year_df = combined_df[combined_df['pub_year'] == end_year]
        current_year_df = current_year_df[current_year_df['study_design'] != 'MR']

        # cumulative counts
        cumulative_counts['p_excitatory'] = (combined_df['relationship'] == 'excitatory').sum()
        cumulative_counts['p_no_change']  = (combined_df['relationship'] == 'no_change').sum()
        cumulative_counts['p_inhibitory'] = (combined_df['relationship'] == 'inhibitory').sum()

        # custom analysis (user-provided)
        analysis_result = analyze_tri_df(combined_df, False)
        curr_loc      = analysis_result.pop('loe')
        curr_biggest  = analysis_result.pop('biggest')

        prev_loc, prev_biggest = curr_loc, curr_biggest

        biggest_p = max(analysis_result.values())
        biggest_p_label = max(analysis_result, key=analysis_result.get)

        results.append({
            'end_year': end_year,
            'loc':      curr_loc,
            **analysis_result,
            'cumulative_counts_p_excitatory': cumulative_counts['p_excitatory'],
            'cumulative_counts_p_no_change':  cumulative_counts['p_no_change'],
            'cumulative_counts_p_inhibitory': cumulative_counts['p_inhibitory'],
            'biggest_p':       biggest_p,
            'biggest_p_label': biggest_p_label
        })

    # ─────────────────────────── prepare plotting df ───────────────────
    results_df = pd.DataFrame(results).sort_values('end_year')
    results_df_disp = results_df[(results_df['end_year'] >= start_display) &
                                 (results_df['end_year'] <= end_display)]

    if source_data_path:
        results_df_disp.to_csv(source_data_path, index=False)

    # ─────────────────────────────── plotting ──────────────────────────
    fig, ax1 = plt.subplots(figsize=(14, 8))
    ax2 = ax1.twinx()

    # right-axis: cumulative counts
    ax2.plot(results_df_disp['end_year'],
             results_df_disp['cumulative_counts_p_excitatory'],
             linestyle='dotted', linewidth=2, color='green',
             label='Excitatory')
    ax2.plot(results_df_disp['end_year'],
             results_df_disp['cumulative_counts_p_no_change'],
             linestyle='dotted', linewidth=2, color='orange',
             label='No Change')
    ax2.plot(results_df_disp['end_year'],
             results_df_disp['cumulative_counts_p_inhibitory'],
             linestyle='dotted', linewidth=2, color='red',
             label='Inhibitory')

    # left-axis: probabilities
    ax1.plot(results_df_disp['end_year'], results_df_disp['p_excitatory'],
             marker='o', color='green',  label='p_excitatory', zorder=3)
    ax1.plot(results_df_disp['end_year'], results_df_disp['p_no_change'],
             marker='o', color='orange', label='p_no_change',  zorder=3)
    ax1.plot(results_df_disp['end_year'], results_df_disp['p_inhibitory'],
             marker='o', color='red',    label='p_inhibitory', zorder=3)

    # axis labels
    ax1.set_xlabel('Year', fontsize=base_fontsize + 2)
    ax1.set_ylabel('Probability of Relation', fontsize=base_fontsize + 2)
    ax2.set_ylabel('Number of Studies', fontsize=base_fontsize + 2)  # ← added label

    ax1.set_ylim(0, 1)

    # 5-year ticks
    start_tick = (results_df_disp['end_year'].min() // 5) * 5
    end_tick   = (results_df_disp['end_year'].max() // 5) * 5
    xticks = np.arange(start_tick, end_tick + 1, 5)
    ax1.set_xticks(xticks)
    ax1.set_xticklabels(xticks, fontsize=base_fontsize)
    ax1.tick_params(axis='y', labelsize=base_fontsize)
    ax2.tick_params(axis='y', labelsize=base_fontsize)
    ax1.set_xlim(results_df_disp['end_year'].min(), results_df_disp['end_year'].max())
    ax1.grid(False)

    # focus year marker
    if focus_year and focus_year in results_df_disp['end_year'].values:
        ax1.axvline(focus_year, linestyle='dashed', linewidth=1.5, color='blue')
        ax1.text(focus_year, 0.53, str(focus_year),
                 color='blue', fontsize=base_fontsize,
                 ha='center', fontweight='bold')

    # # legends
    # ax1.legend(loc='upper left',  fontsize=base_fontsize)
    #ax2.legend(loc='upper center', fontsize=base_fontsize)
    legend_handles = [
    Patch(facecolor='green',  edgecolor='green',  label='Excitatory'),
    Patch(facecolor='orange', edgecolor='orange', label='No Change'),
    Patch(facecolor='red',    edgecolor='red',    label='Inhibitory')]

# Add it once, on whichever axis you prefer (ax1 here)
    ax1.legend(handles=legend_handles,
              loc='upper left',     # pick a position you like
              fontsize=base_fontsize)

    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight')

    plt.show()