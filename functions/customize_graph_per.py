def customize_graph_percentage(ax, plot_type='bar'):
    # Set background colors for the figure and axes
    fig = ax.get_figure()
    fig.patch.set_facecolor('#0F1116') 
    ax.set_facecolor('#0F1116') 

    # Change color of all elements to white
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    ax.title.set_color('white')

    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')

    for spine in ax.spines.values():
        spine.set_color('white')

    # Get the heights of the bars and set the y-axis limits accordingly
    if plot_type == 'bar' or plot_type == 'barh':
        bar_heights = [bar.get_height() for bar in ax.patches]
        if bar_heights:
            max_height = max(bar_heights)
            min_height = min(bar_heights)

            # Calculate the 10% margin based on the range between max and min heights
            range_height = max_height - min_height
            y_lower = min_height - (range_height * 0.10)  # 10% below the range
            y_upper = max_height + (range_height * 0.10)  # 10% above the range

            # Set the y-axis limits based on the adjusted lower and upper bounds
            ax.set_ylim(y_lower, y_upper)

        y_min, y_max = ax.get_ylim()
        y_range = y_max - y_min

    # Calculate total count of bars
    total_counts = sum([p.get_height() for p in ax.patches])

    # Add bar values for bar and barh as percentage
    if plot_type == 'bar':
        for bar in ax.patches:
            yval = bar.get_height()
            percentage = (yval / total_counts) * 100
            # Add text above each bar showing its height as a percentage of the total count
            ax.text(bar.get_x() + bar.get_width() / 2, yval + y_range * 0.02, f'{round(percentage, 2)}%', 
                    ha='center', va='bottom', color='white', fontweight='bold')

    elif plot_type == 'barh':
        for bar in ax.patches:
            xval = bar.get_width()
            percentage = (xval / total_counts) * 100
            # Add text inside each horizontal bar showing its width as a percentage of the total count
            ax.text(xval / 2, bar.get_y() + bar.get_height() / 2, f'{round(percentage, 2)}%', 
                    ha='center', va='center', color='white', fontweight='bold')
