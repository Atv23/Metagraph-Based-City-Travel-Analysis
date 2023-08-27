def your_program_function(inp_data):
    # Your code logic here
    # Process the input data and generate the output
    import metagraph as mg
    import pandas as pd
    import networkx as nx
    
    RAW_DATA_CSV = inp_data 
    data_df = pd.read_csv(RAW_DATA_CSV, encoding="ISO-8859-1")
    
    RELEVANT_COLUMNS = [
        'PASSENGERS',
        'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME',
        'DEST_AIRPORT_ID',   'DEST_AIRPORT_SEQ_ID',   'DEST_CITY_MARKET_ID',   'DEST',   'DEST_CITY_NAME',
    ]
    relevant_df = data_df[RELEVANT_COLUMNS]
    relevant_df = relevant_df[relevant_df.PASSENGERS != 0.0]
    
    passenger_flow_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID', 'PASSENGERS']]
    pd.set_option('display.max_rows', None)
    
    passenger_flow_df = passenger_flow_df.groupby(['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID']) \
                            .PASSENGERS.sum() \
                            .reset_index()
    
    passenger_flow_df['INVERSE_PASSENGER_COUNT'] = passenger_flow_df.PASSENGERS.map(lambda passenger_count: 1/passenger_count)
    
    assert len(passenger_flow_df[passenger_flow_df.INVERSE_PASSENGER_COUNT != passenger_flow_df.INVERSE_PASSENGER_COUNT]) == 0, "Edge list has NaN weights."
    
    passenger_flow_edge_map = mg.wrappers.EdgeMap.PandasEdgeMap(
        passenger_flow_df,
        'ORIGIN_CITY_MARKET_ID',
        'DEST_CITY_MARKET_ID',
        'INVERSE_PASSENGER_COUNT',
        is_directed=True
    )
    
    # Build the metagraph from the EdgeMap
    
    origin_city_market_id_info_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME']] \
                                        .rename(columns={'ORIGIN_CITY_MARKET_ID': 'CITY_MARKET_ID',
                                                         'ORIGIN': 'AIRPORT',
                                                         'ORIGIN_CITY_NAME': 'CITY_NAME'})
    dest_city_market_id_info_df = relevant_df[['DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME']] \
                                        .rename(columns={'DEST_CITY_MARKET_ID': 'CITY_MARKET_ID',
                                                         'DEST': 'AIRPORT',
                                                         'DEST_CITY_NAME': 'CITY_NAME'})
    passenger_flow_graph = mg.algos.util.graph.build(passenger_flow_edge_map)
    
    city_market_id_info_df = pd.concat([origin_city_market_id_info_df, 
    dest_city_market_id_info_df]).set_index('CITY_MARKET_ID').sort_values('CITY_MARKET_ID',ascending=True)
    
    city_market_id_info_df = city_market_id_info_df.groupby('CITY_MARKET_ID').agg({'AIRPORT': set, 'CITY_NAME': set})
    
    betweenness_centrality = mg.algos.centrality.betweenness(passenger_flow_graph, normalize=False)
    
    best_betweenness_centrality_node_vector = mg.algos.util.nodemap.sort(betweenness_centrality, ascending=False)
    
    best_betweenness_centrality_node_set = mg.algos.util.nodeset.from_vector(best_betweenness_centrality_node_vector)
    
    best_betweenness_centrality_node_to_score_map = mg.algos.util.nodemap.select(betweenness_centrality, best_betweenness_centrality_node_set)
    
    best_betweenness_centrality_node_to_score_map = mg.translate(best_betweenness_centrality_node_to_score_map, mg.types.NodeMap.PythonNodeMapType)
    
    best_betweenness_centrality_scores_df = pd.DataFrame(best_betweenness_centrality_node_to_score_map.items()).rename(columns={0:'CITY_MARKET_ID', 1:'BETWEENNESS_CENTRALITY_SCORE'}).set_index('CITY_MARKET_ID')
    best_betweenness_centrality_scores_df.sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False)
    
    final_data= pd.DataFrame(best_betweenness_centrality_scores_df.join(city_market_id_info_df).sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False))
    final_data

    output_data = final_data
    output_data.to_csv('output.csv', index=False)
    return output_data

def bar_func(inp_data):
    import metagraph as mg
    import pandas as pd

    RAW_DATA_CSV = inp_data
    raw_data_df = pd.read_csv(RAW_DATA_CSV)

    RELEVANT_COLUMNS = [
        'PASSENGERS',
        'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME',
        'DEST_AIRPORT_ID',   'DEST_AIRPORT_SEQ_ID',   'DEST_CITY_MARKET_ID',   'DEST',   'DEST_CITY_NAME', 
    ]
    relevant_df = raw_data_df[RELEVANT_COLUMNS]
    relevant_df = relevant_df[relevant_df.PASSENGERS != 0.0]

    passenger_flow_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID', 'PASSENGERS']]
    pd.set_option('display.max_rows', None)

    passenger_flow_df = passenger_flow_df.groupby(['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID']) \
                            .PASSENGERS.sum() \
                            .reset_index()

    passenger_flow_df['INVERSE_PASSENGER_COUNT'] = passenger_flow_df.PASSENGERS.map(lambda passenger_count: 1/passenger_count)

    assert len(passenger_flow_df[passenger_flow_df.INVERSE_PASSENGER_COUNT != passenger_flow_df.INVERSE_PASSENGER_COUNT]) == 0, "Edge list has NaN weights."

    passenger_flow_edge_map = mg.wrappers.EdgeMap.PandasEdgeMap(
        passenger_flow_df,
        'ORIGIN_CITY_MARKET_ID',
        'DEST_CITY_MARKET_ID',
        'INVERSE_PASSENGER_COUNT',
        is_directed=True
    )

    # Build the metagraph from the EdgeMap
    passenger_flow_graph = mg.algos.util.graph.build(passenger_flow_edge_map)

    origin_city_market_id_info_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME']] \
                                        .rename(columns={'ORIGIN_CITY_MARKET_ID': 'CITY_MARKET_ID',
                                                         'ORIGIN': 'AIRPORT',
                                                         'ORIGIN_CITY_NAME': 'CITY_NAME'})
    dest_city_market_id_info_df = relevant_df[['DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME']] \
                                        .rename(columns={'DEST_CITY_MARKET_ID': 'CITY_MARKET_ID',
                                                         'DEST': 'AIRPORT',
                                                         'DEST_CITY_NAME': 'CITY_NAME'})

    city_market_id_info_df = pd.concat([origin_city_market_id_info_df, dest_city_market_id_info_df]).set_index('CITY_MARKET_ID').sort_values('CITY_MARKET_ID',ascending=True)

    city_market_id_info_df = city_market_id_info_df.groupby('CITY_MARKET_ID').agg({'AIRPORT': set, 'CITY_NAME': set})

    betweenness_centrality = mg.algos.centrality.betweenness(passenger_flow_graph, normalize=False)

    best_betweenness_centrality_node_vector = mg.algos.util.nodemap.sort(betweenness_centrality, ascending=False)

    best_betweenness_centrality_node_set = mg.algos.util.nodeset.from_vector(best_betweenness_centrality_node_vector)

    best_betweenness_centrality_node_to_score_map = mg.algos.util.nodemap.select(betweenness_centrality, best_betweenness_centrality_node_set)

    best_betweenness_centrality_node_to_score_map = mg.translate(best_betweenness_centrality_node_to_score_map, mg.types.NodeMap.PythonNodeMapType)

    best_betweenness_centrality_scores_df = pd.DataFrame(best_betweenness_centrality_node_to_score_map.items()).rename(columns={0:'CITY_MARKET_ID', 1:'BETWEENNESS_CENTRALITY_SCORE'}).set_index('CITY_MARKET_ID')

    best_betweenness_centrality_scores_df.sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False)

    final_data= pd.DataFrame(best_betweenness_centrality_scores_df.join(city_market_id_info_df).sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False))

    import pandas as pd

    # Create an empty dictionary
    score_airport_dict = {}

    # Iterate over the rows of the final_data DataFrame
    for index, row in final_data.head(10).iterrows():
        score = row['BETWEENNESS_CENTRALITY_SCORE']
        airport = row['AIRPORT']

        # Check if the score already exists as a key in the dictionary
        if score in score_airport_dict:
            # Check if the airport is not already present in the list
            if airport not in score_airport_dict[score]:
                score_airport_dict[score].append(airport)
        else:
            # Create a new key-value pair with the score and the airport as a list
            score_airport_dict[score] = [airport]
    import matplotlib.pyplot as plt
    import numpy as np

    # Flatten the sets of airports and scores into separate lists
    airports = []
    scores = []
    for score, airport_set in score_airport_dict.items():
        airports.extend(list(airport_set))
        scores.extend([score] * len(airport_set))

    # Create unique colors for each airport
    color_map = plt.get_cmap('rainbow')
    colors = [color_map(i) for i in np.linspace(0, 1, len(airports))]

    # Create the bar graph
    plt.bar(range(len(airports)), scores, color=colors)

    # Set the x-axis labels to the airports
    plt.xticks(range(len(airports)), airports, rotation=90)

    # Set the labels and title
    plt.xlabel('Airports')
    plt.ylabel('BETWEENNESS_CENTRALITY_SCORE')
    plt.title('BETWEENNESS_CENTRALITY_SCORE of Airports')

    # Sort the scores and colors together based on scores
    sorted_indices = np.argsort(scores)
    sorted_scores = [scores[i] for i in sorted_indices]
    sorted_colors = [colors[i] for i in sorted_indices]

    # Create a custom legend for the scores and colors
    legend_labels = sorted(set(scores))
    legend_handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=str(l)) for l, c in zip(legend_labels, sorted_colors)]

    # Add the legend to the plot
    plt.legend(handles=legend_handles, title='BETWEENNESS_CENTRALITY_SCORE', loc='upper right')

    # Display the graph
    # plt.show()

        
    return plt.gcf()

def line_func(inp_data):
    import metagraph as mg
    import pandas as pd

    RAW_DATA_CSV = inp_data # https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=258
    raw_data_df = pd.read_csv(RAW_DATA_CSV)

    RELEVANT_COLUMNS = [
        'PASSENGERS',
        'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME', 
        'DEST_AIRPORT_ID',   'DEST_AIRPORT_SEQ_ID',   'DEST_CITY_MARKET_ID',   'DEST',   'DEST_CITY_NAME',
    ]
    relevant_df = raw_data_df[RELEVANT_COLUMNS]
    relevant_df = relevant_df[relevant_df.PASSENGERS != 0.0]

    passenger_flow_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID', 'PASSENGERS']]
    pd.set_option('display.max_rows', None)

    passenger_flow_df = passenger_flow_df.groupby(['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID']) \
                            .PASSENGERS.sum() \
                            .reset_index()

    passenger_flow_df['INVERSE_PASSENGER_COUNT'] = passenger_flow_df.PASSENGERS.map(lambda passenger_count: 1/passenger_count)

    assert len(passenger_flow_df[passenger_flow_df.INVERSE_PASSENGER_COUNT != passenger_flow_df.INVERSE_PASSENGER_COUNT]) == 0, "Edge list has NaN weights."

    passenger_flow_edge_map = mg.wrappers.EdgeMap.PandasEdgeMap(
        passenger_flow_df,
        'ORIGIN_CITY_MARKET_ID',
        'DEST_CITY_MARKET_ID',
        'INVERSE_PASSENGER_COUNT',
        is_directed=True
    )

    # Build the metagraph from the EdgeMap
    passenger_flow_graph = mg.algos.util.graph.build(passenger_flow_edge_map)

    origin_city_market_id_info_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME']] \
                                        .rename(columns={'ORIGIN_CITY_MARKET_ID': 'CITY_MARKET_ID',
                                                         'ORIGIN': 'AIRPORT',
                                                         'ORIGIN_CITY_NAME': 'CITY_NAME'})
    dest_city_market_id_info_df = relevant_df[['DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME']] \
                                        .rename(columns={'DEST_CITY_MARKET_ID': 'CITY_MARKET_ID',
                                                         'DEST': 'AIRPORT',
                                                         'DEST_CITY_NAME': 'CITY_NAME'})

    city_market_id_info_df = pd.concat([origin_city_market_id_info_df, dest_city_market_id_info_df]).set_index('CITY_MARKET_ID').sort_values('CITY_MARKET_ID',ascending=True)

    city_market_id_info_df = city_market_id_info_df.groupby('CITY_MARKET_ID').agg({'AIRPORT': set, 'CITY_NAME': set})

    betweenness_centrality = mg.algos.centrality.betweenness(passenger_flow_graph, normalize=False)

    best_betweenness_centrality_node_vector = mg.algos.util.nodemap.sort(betweenness_centrality, ascending=False)

    best_betweenness_centrality_node_set = mg.algos.util.nodeset.from_vector(best_betweenness_centrality_node_vector)

    best_betweenness_centrality_node_to_score_map = mg.algos.util.nodemap.select(betweenness_centrality, best_betweenness_centrality_node_set)

    best_betweenness_centrality_node_to_score_map = mg.translate(best_betweenness_centrality_node_to_score_map, mg.types.NodeMap.PythonNodeMapType)

    best_betweenness_centrality_scores_df = pd.DataFrame(best_betweenness_centrality_node_to_score_map.items()).rename(columns={0:'CITY_MARKET_ID', 1:'BETWEENNESS_CENTRALITY_SCORE'}).set_index('CITY_MARKET_ID')

    best_betweenness_centrality_scores_df.sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False)

    final_data= pd.DataFrame(best_betweenness_centrality_scores_df.join(city_market_id_info_df).sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False))

    import pandas as pd

    # Create an empty dictionary
    score_airport_dict = {}

    # Iterate over the rows of the final_data DataFrame
    for index, row in final_data.head(10).iterrows():
        score = row['BETWEENNESS_CENTRALITY_SCORE']
        airport = row['AIRPORT']

        # Check if the score already exists as a key in the dictionary
        if score in score_airport_dict:
            # Check if the airport is not already present in the list
            if airport not in score_airport_dict[score]:
                score_airport_dict[score].append(airport)
        else:
            # Create a new key-value pair with the score and the airport as a list
            score_airport_dict[score] = [airport]
            
    import matplotlib.pyplot as plt
    import numpy as np

    # Flatten the sets of airports and scores into separate lists
    airports = []
    scores = []
    for score, airport_set in score_airport_dict.items():
        airports.extend(list(airport_set))
        scores.extend([score] * len(airport_set))

    # Sort the airports and scores based on scores
    sorted_indices = np.argsort(scores)
    sorted_airports = [', '.join(sorted(list(airport_set))) for airport_set in np.array(airports)[sorted_indices]]
    sorted_scores = [scores[i] for i in sorted_indices]

    # Reverse the order of sorted airports and scores
    sorted_airports = sorted_airports[::-1]
    sorted_scores = sorted_scores[::-1]

    # Create a line graph
    plt.plot(sorted_airports, sorted_scores, marker='o')

    # Set the x-axis labels to the airports
    plt.xticks(rotation=90)

    # Set the labels and title
    plt.xlabel('Airports')
    plt.ylabel('BETWEENNESS_CENTRALITY_SCORE')
    plt.title('BETWEENNESS_CENTRALITY_SCORE of Airports')

    # Display the graph
    plt.show()
            
    return plt.gcf()

def metagraph_func(inp_data):
    import metagraph as mg
    import pandas as pd

    RAW_DATA_CSV = inp_data # https://www.transtats.bts.gov/DL_SelectFields.asp?Table_ID=258
    raw_data_df = pd.read_csv(RAW_DATA_CSV)

    RELEVANT_COLUMNS = [
        'PASSENGERS',
        'ORIGIN_AIRPORT_ID', 'ORIGIN_AIRPORT_SEQ_ID', 'ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME',
        'DEST_AIRPORT_ID',   'DEST_AIRPORT_SEQ_ID',   'DEST_CITY_MARKET_ID',   'DEST',   'DEST_CITY_NAME',
    ]
    relevant_df = raw_data_df[RELEVANT_COLUMNS]
    relevant_df = relevant_df[relevant_df.PASSENGERS != 0.0]

    passenger_flow_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID', 'PASSENGERS']]
    pd.set_option('display.max_rows', None)

    passenger_flow_df = passenger_flow_df.groupby(['ORIGIN_CITY_MARKET_ID', 'DEST_CITY_MARKET_ID']) \
                            .PASSENGERS.sum() \
                            .reset_index()

    passenger_flow_df['INVERSE_PASSENGER_COUNT'] = passenger_flow_df.PASSENGERS.map(lambda passenger_count: 1/passenger_count)

    assert len(passenger_flow_df[passenger_flow_df.INVERSE_PASSENGER_COUNT != passenger_flow_df.INVERSE_PASSENGER_COUNT]) == 0, "Edge list has NaN weights."

    passenger_flow_edge_map = mg.wrappers.EdgeMap.PandasEdgeMap(
        passenger_flow_df,
        'ORIGIN_CITY_MARKET_ID',
        'DEST_CITY_MARKET_ID',
        'INVERSE_PASSENGER_COUNT',
        is_directed=True
    )

    # Build the metagraph from the EdgeMap
    passenger_flow_graph = mg.algos.util.graph.build(passenger_flow_edge_map)

    origin_city_market_id_info_df = relevant_df[['ORIGIN_CITY_MARKET_ID', 'ORIGIN', 'ORIGIN_CITY_NAME']] \
                                        .rename(columns={'ORIGIN_CITY_MARKET_ID': 'CITY_MARKET_ID',
                                                         'ORIGIN': 'AIRPORT',
                                                         'ORIGIN_CITY_NAME': 'CITY_NAME'})
    dest_city_market_id_info_df = relevant_df[['DEST_CITY_MARKET_ID', 'DEST', 'DEST_CITY_NAME']] \
                                        .rename(columns={'DEST_CITY_MARKET_ID': 'CITY_MARKET_ID',
                                                         'DEST': 'AIRPORT',
                                                         'DEST_CITY_NAME': 'CITY_NAME'})

    city_market_id_info_df = pd.concat([origin_city_market_id_info_df, dest_city_market_id_info_df]).set_index('CITY_MARKET_ID').sort_values('CITY_MARKET_ID',ascending=True)

    city_market_id_info_df = city_market_id_info_df.groupby('CITY_MARKET_ID').agg({'AIRPORT': set, 'CITY_NAME': set})

    betweenness_centrality = mg.algos.centrality.betweenness(passenger_flow_graph, normalize=False)

    best_betweenness_centrality_node_vector = mg.algos.util.nodemap.sort(betweenness_centrality, ascending=False)

    best_betweenness_centrality_node_set = mg.algos.util.nodeset.from_vector(best_betweenness_centrality_node_vector)

    best_betweenness_centrality_node_to_score_map = mg.algos.util.nodemap.select(betweenness_centrality, best_betweenness_centrality_node_set)

    best_betweenness_centrality_node_to_score_map = mg.translate(best_betweenness_centrality_node_to_score_map, mg.types.NodeMap.PythonNodeMapType)

    best_betweenness_centrality_scores_df = pd.DataFrame(best_betweenness_centrality_node_to_score_map.items()).rename(columns={0:'CITY_MARKET_ID', 1:'BETWEENNESS_CENTRALITY_SCORE'}).set_index('CITY_MARKET_ID')

    best_betweenness_centrality_scores_df.sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False)

    final_data= pd.DataFrame(best_betweenness_centrality_scores_df.join(city_market_id_info_df).sort_values('BETWEENNESS_CENTRALITY_SCORE', ascending=False))

    import pandas as pd

    # Create an empty dictionary
    score_airport_dict = {}

    # Iterate over the rows of the final_data DataFrame
    for index, row in final_data.head(10).iterrows():
        score = row['BETWEENNESS_CENTRALITY_SCORE']
        airport = row['AIRPORT']

        # Check if the score already exists as a key in the dictionary
        if score in score_airport_dict:
            # Check if the airport is not already present in the list
            if airport not in score_airport_dict[score]:
                score_airport_dict[score].append(airport)
        else:
            # Create a new key-value pair with the score and the airport as a list
            score_airport_dict[score] = [airport]
            
    import networkx as nx
    import matplotlib.pyplot as plt

    # Create an empty metagraph
    mg = nx.Graph()

    # Iterate over the city_market_ids and their corresponding BETWEENNESS_CENTRALITY_SCORE
    for index, row in best_betweenness_centrality_scores_df.iterrows():
        city_market_id = index
        score = row['BETWEENNESS_CENTRALITY_SCORE']

        # Check if the score exists as a key in the score_airport_dict
        if score in score_airport_dict:
            # Create a subgraph for airports with the same BETWEENNESS_CENTRALITY_SCORE
            subgraph = nx.Graph()
            airports = score_airport_dict[score]

            # Add individual airports as nodes to the subgraph
            for airport_set in airports:
                airport_tuple = tuple(airport_set)
                subgraph.add_node(airport_tuple)

            # Add the subgraph as a node to the metagraph
            mg.add_node(city_market_id, graph=subgraph, score=score)

    # Connect city_market_id nodes in the metagraph with edges weighted by the BETWEENNESS_CENTRALITY_SCORE
    for city_market_id1, data1 in mg.nodes(data=True):
        for city_market_id2, data2 in mg.nodes(data=True):
            if city_market_id1 != city_market_id2:
                score1 = data1['score']
                score2 = data2['score']

                # Calculate the weight based on the absolute difference between the scores
                weight = abs(score1 - score2)

                # Add an edge with the weight equal to the difference in scores
                mg.add_edge(city_market_id1, city_market_id2, weight=weight)

    # Draw the metagraph
    pos = nx.spring_layout(mg, seed=42)
    plt.figure(figsize=(12, 8))
    nx.draw_networkx(mg, pos, with_labels=True, node_color='#00FFEF', node_size=500, font_size=8)
    edge_labels = nx.get_edge_attributes(mg, 'weight')
    nx.draw_networkx_edge_labels(mg, pos, edge_labels=edge_labels)
    plt.title('Metagraph Visualization')
    plt.show()
    return plt.gcf()