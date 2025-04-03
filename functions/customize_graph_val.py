def customize_graph(ax, plot_type='bar'):
    # Get the figure object associated with the Axes
    fig = ax.get_figure()
    # Set the figure background color
    fig.patch.set_facecolor('#0F1116') 
    # Set the Axes background color
    ax.set_facecolor('#0F1116')

    # Change the color of the x-axis label, y-axis label, and title to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')
    # Change the color of the x-axis and y-axis ticks to white
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    # Change the color of the axes spines to white
    for spine in ax.spines.values():
        spine.set_color('white')

    # Customizations for bar plots
    if plot_type == 'bar':
        # Get the heights of all bars in the bar plot
        bar_heights = [bar.get_height() for bar in ax.patches]
        if bar_heights:
            # Determine the maximum and minimum heights of the bars
            max_height = max(bar_heights)
            min_height = min(bar_heights)

            # Calculate the 10% margin based on the range between max and min heights
            range_height = max_height - min_height
            y_lower = min_height - (range_height * 0.10)  # 10% below the range
            y_upper = max_height + (range_height * 0.10)  # 10% above the range

            # Set the y-axis limits based on the adjusted lower and upper bounds
            ax.set_ylim(y_lower, y_upper)

        # Add a text label above each bar indicating its height
        for bar in ax.patches:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval + (range_height * 0.02), round(yval, 2), 
                    ha='center', va='bottom', color='white', fontweight='bold')

    # Customizations for horizontal bar plots
    elif plot_type == 'barh':
        # Get the widths of all bars in the bar plot
        bar_widths = [bar.get_width() for bar in ax.patches]
        if bar_widths:
            # Determine the maximum and minimum widths of the bars
            max_width = max(bar_widths)
            min_width = min(bar_widths)
            # Set the x-axis limits to include a 10% margin to the left and right of the bar widths
            ax.set_xlim(min_width * 0.90, max_width * 1.10)
        # Get the current x-axis limits
        x_min, x_max = ax.get_xlim()
        # Calculate the range of x-axis values
        x_range = x_max - x_min

        # Add a text label to the right of each bar indicating its width
        for bar in ax.patches:
            xval = bar.get_width()
            ax.text(xval + x_range * 0.02, bar.get_y() + bar.get_height() / 2, round(xval, 2), 
                    ha='left', va='center', color='white', fontweight='bold')
