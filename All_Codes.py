import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns


class homework:


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------Adatbeolvasás és szűrés---------------------------------------------------------------------------   
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def __init__(self):
        self.soc_pokec_profiles = pd.read_csv('soc-pokec-profiles.txt', index_col= False, sep="\t")
        self.soc_pokec_relationships = pd.read_csv('soc-pokec-relationships.txt', index_col= False, sep="\t")

    def szures_profil(self, soc_pokec_profiles):
        soc_pokec_profiles = soc_pokec_profiles[['1', '1.1', '1.2','26']]
        new_row = pd.DataFrame({'1':1, '1.1':1.0, '1.2':1.0, '26':26}, index = [0])
        soc_pokec_profiles = pd.concat([new_row, soc_pokec_profiles.loc[:]]).reset_index(drop = True)
        soc_pokec_profiles = soc_pokec_profiles.rename(columns = {"1":"User_id", "1.1":"Public" ,"1.2":"Gender", "26":"Age"})
        all_profiles = soc_pokec_profiles
        return all_profiles
        
    def szures_retationships(self, soc_pokec_relationships):
        soc_pokec_relationships = soc_pokec_relationships.rename(columns = {"1":"Source", "13":"Sink"})
        all_edges = soc_pokec_relationships
        return all_edges
    
    def select_relevant_profiles(self, all_profiles):
        """Releváns profilok kiválasztása
        Kritérium:
        * publikus
        * 14 év feletti életkor
        * nem legyen megadva
        """
        public_condition = all_profiles["Public"] == 1
        age_condition = all_profiles["Age"] > 14
        gender_condition = all_profiles["Gender"].isin([0, 1])
        relevant_profiles = all_profiles.loc[public_condition & age_condition & gender_condition].reset_index(drop = True)
        selected_ids = relevant_profiles["User_id"].unique()
        return relevant_profiles, selected_ids

    def select_relevant_edges(self, all_edges, selected_ids):
        """Élek adatbázisból relevánsak kiválasztása az előző kritériumok szerint megtisztított releváns profilok alapján"""
        source_condition = all_edges["Source"].isin(selected_ids)
        sink_condition = all_edges["Sink"].isin(selected_ids)
        relevant_edges = all_edges.loc[source_condition & sink_condition].reset_index(drop = True)
        return relevant_edges
    
    def convert_edges_to_undirected(self, relevant_edges):
        """Irány nélküli éleké konvertlás és csak az oda-vissza kapcsolatok megtartása"""
        undirected_edges = (
            relevant_edges.assign(
                Smaller_id=lambda df: df[["Source", "Sink"]].min(axis=1),
                Greater_id=lambda df: df[["Source", "Sink"]].max(axis=1),
            )
            .groupby(["Smaller_id", "Greater_id"])
            .agg({"Source": "count"})
        )
        print(undirected_edges["Source"].value_counts())
        return (
            undirected_edges.loc[undirected_edges["Source"] == 2]
            .drop("Source", axis=1)
            .reset_index()
        )

    def add_node_features_to_edges(self, nodes, edges):
        """Hőtérképek miatt a csúcsok jellemzőinek hozzárendelése az élekhez"""
        edges_w_features = edges.merge(
            nodes[["User_id", "Age", "Gender"]].set_index("User_id"),
            how="left",
            left_on="Smaller_id",
            right_index=True,
        )
        edges_w_features = edges_w_features.merge(
            nodes[["User_id", "Age", "Gender"]].set_index("User_id"),
            how="left",
            left_on="Greater_id",
            right_index=True,
        )
        return edges_w_features
    
        return nodes
    
    def load_and_select_profiles_and_edges(self):
        """load and select relevant profiles, then filter and undirect edges"""
        print("loading profiles")
        soc_pokec_profiles = pd.read_csv('soc-pokec-profiles.txt', index_col= False, sep="\t")
        all_profiles = self.szures_profil(soc_pokec_profiles)
        print("loading edges")
        soc_pokec_relationships = pd.read_csv('soc-pokec-relationships.txt', index_col= False, sep="\t")
        all_edges = self.szures_retationships(soc_pokec_relationships)
        
        relevant_profiles, selected_ids = self.select_relevant_profiles(all_profiles)
        relevant_edges = self.select_relevant_edges(all_edges, selected_ids)

        undirected_edges = self.convert_edges_to_undirected(relevant_edges)
        nodes_with_edges = set(undirected_edges["Smaller_id"].unique()).union(
            undirected_edges["Greater_id"].unique()
        )
        print(f"Selected profiles: {len(relevant_profiles)}")
        print(f"Nodes with edges: {len(nodes_with_edges)}")
        selected_profiles = relevant_profiles[
            relevant_profiles["User_id"].isin(nodes_with_edges)
        ]
        selected_profiles["Age"] = selected_profiles["Age"].clip(upper=50)
        edges_w_features = self.add_node_features_to_edges(selected_profiles, undirected_edges)
        return selected_profiles, undirected_edges, edges_w_features


#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------Gráfépítés-------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------


    def create_graph_from_nodes_and_edges(self, nodes, edges):
        """Networkx package segítségével gráf megalkotása a releváns jellemzők szerint"""
        node_attributes = nodes.set_index("User_id").to_dict(orient="index")
        node_attributes_list = [
            (index, attr_dict) for index, attr_dict in node_attributes.items()
        ]
        G = nx.Graph()
        G.add_nodes_from(node_attributes_list)
        G.add_edges_from(edges.values.tolist())
        return G
    
    
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------Adatvizualizáció: Alapok-------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------  
    
    
    def plot_degree_distribution(self, G):
        """Fokszám-eloszlás ábrázolása"""
        plot_df = (
            pd.Series(dict(G.degree)).value_counts().sort_index().to_frame().reset_index()
        )
        plot_df.columns = ["k", "count"]
        plot_df["log_k"] = np.log(plot_df["k"])
        plot_df["log_count"] = np.log(plot_df["count"])
        fig, ax = plt.subplots()

        ax.scatter(plot_df["k"], plot_df["count"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        fig.suptitle("Mutual Degree Distribution")
        ax.set_xlabel("k")
        ax.set_ylabel("count_k")
        
    def plot_age_distribution_by_gender(self, nodes):
        """Hisztogram nemenként"""
        plot_df = nodes[["Age", "Gender"]].copy(deep=True).astype(float)
        plot_df["Gender"] = plot_df["Gender"].replace({0.0: "woman", 1.0: "man"})
        sns.histplot(data=plot_df, x="Age", hue="Gender", bins=np.arange(0, 45, 5) + 15)
        
        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------Adatvizualizáció: Figure 3-----------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------    
       
    
    def plot_node_degree_by_gender(self, nodes, G):
        """Nemenként és koronként az átlagos fokszám ábrázolása"""
        nodes_w_degree = nodes.set_index("User_id").merge(
            pd.Series(dict(G.degree)).to_frame(),
            how="left",
            left_index=True,
            right_index=True,
        )
        nodes_w_degree = nodes_w_degree.rename({0: "degree"}, axis=1)
        nodes_w_degree["Gender"].replace([0.0, 1.0], ["Female", "Male"], inplace=True)
        plot_df = (
            nodes_w_degree.groupby(["Age", "Gender"]).agg({"degree": "mean"}).reset_index()
        )

        ax = sns.lineplot(
            data=plot_df, x="Age", y="degree", hue="Gender", palette=["red", "blue"]
        )
        ax.set_xlabel("Age")
        ax.set_ylabel("Degree")
        ax.set_title("(a) Degree Centrality", y = -0.25)
        
    def plot_neighbour_conn_by_gender(self, nodes, G):
        """Neighbour-connectivity ábrázolása: minden egyes csúcs szomszédjainak átlagos fokszáma"""
        nodes_w_neighbor_conn = nodes

        #Networkx package average_neighbor_degree függvény használata 
        nodes_w_neighbor_conn = nodes_w_neighbor_conn.assign(
            neighbor_conn=nodes_w_neighbor_conn.User_id.map(nx.average_neighbor_degree(G))
        )
        nodes_w_neighbor_conn["Gender"].replace(
            [0.0, 1.0], ["Female", "Male"], inplace=True
        )

        plot_df = (
            nodes_w_neighbor_conn.groupby(["Age", "Gender"])
            .agg({"neighbor_conn": "mean"})
            .reset_index()
        )
        ax = sns.lineplot(
            data=plot_df, x="Age", y="neighbor_conn", hue="Gender", palette=["red", "blue"]
        )
        ax.set_xlabel("Age")
        ax.set_ylabel("Neighbour Connectivity")
        ax.set_title("(b) Neighbour Connectivity", y = -0.25)
        
    def plot_triadic_clos_by_gender(self, nodes, G):
        """Triadic closure ábrázolása: minden egyes csúcs lokális klaszterkoefficiense"""
        nodes_w_triadic_clos = nodes
        nodes_w_triadic_clos = nodes_w_triadic_clos.assign(
            triadic_clos=nodes_w_triadic_clos.User_id.map(nx.clustering(G))
        )
        nodes_w_triadic_clos["Gender"].replace([0.0, 1.0], ["Female", "Male"], inplace=True)

        plot_df = (
            nodes_w_triadic_clos.groupby(["Age", "Gender"])
            .agg({"triadic_clos": "mean"})
            .reset_index()
        )
        ax = sns.lineplot(
            data=plot_df, x="Age", y="triadic_clos", hue="Gender", palette=["red", "blue"]
        )
        ax.set_xlabel("Age")
        ax.set_ylabel("cc")
        ax.set_title("(c) Triadic Closure", y = -0.25)
        
    def figure3_plot(self, nodes, G):
        """Figure 3 ábrázolása"""
        plt.figure(figsize=(14, 4))
        plt.subplot(1, 3, 1)
        self.plot_node_degree_by_gender(selected_profiles, G)
        plt.subplot(1, 3, 2)
        self.plot_neighbour_conn_by_gender(selected_profiles, G)
        plt.subplot(1, 3, 3)
        self.plot_triadic_clos_by_gender(selected_profiles, G)

        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------------------------Adatvizualizáció: Figure 5------------------------------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------    
       
        
    def plot_age_relations_heatmap(self, edges_w_features):
        """Hőtérkép a nemenkénti kapcsolatok arányaira az életkor függvényében"""
        plot_df = edges_w_features.groupby(["Gender_x", "Gender_y", "Age_x", "Age_y"]).agg(
            {"Smaller_id": "count"}
        )
        plot_df_w_w = plot_df.loc[(0, 0)].reset_index()
        plot_df_heatmap = plot_df_w_w.pivot_table(
            index="Age_x", columns="Age_y", values="Smaller_id"
        ).fillna(0)
        plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
        ax = sns.heatmap(plot_df_heatmap_logged, cmap = "jet")
        ax.invert_yaxis()
        ax.set_xlabel("Age")
        ax.set_ylabel("Age")
        ax.set_title("(a) #connections per pair", y = -0.15)
        
    def plot_age_relations_heatmap_M_M(self, edges_w_features):
        """Hőtérkép a férfi-férfi kapcsolatok arányaira az életkor függvényében"""

        # Férfi-férfi kapcsolatok kiszűrése
        edges_w_features_M_M = edges_w_features.loc[
            (edges_w_features["Gender_x"] == 1.0) & (edges_w_features["Gender_y"] == 1.0)
        ]

        plot_df = edges_w_features_M_M.groupby(
            ["Gender_x", "Gender_y", "Age_x", "Age_y"]
        ).agg({"Smaller_id": "count"})
        # plot_df_w_w = plot_df.loc[(0, 0)].reset_index() -> not needed
        plot_df_heatmap = plot_df.pivot_table(
            index="Age_x", columns="Age_y", values="Smaller_id"
        ).fillna(0)
        plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
        ax = sns.heatmap(plot_df_heatmap_logged, cmap="jet")
        ax.invert_yaxis()
        ax.set_xlabel("Age (Male)")
        ax.set_ylabel("Age (Male)")
        ax.set_title("(B) #connections per M-M pair", y = -0.15)
        
    def plot_age_relations_heatmap_F_F(self, edges_w_features):
        """Hőtérkép a nő-nő kapcsolatok arányaira az életkor függvényében"""

        # Nő-nő kapcsolatok kiszűrése
        edges_w_features_F_F = edges_w_features.loc[
            (edges_w_features["Gender_x"] == 0.0) & (edges_w_features["Gender_y"] == 0.0)
        ]

        plot_df = edges_w_features_F_F.groupby(
            ["Gender_x", "Gender_y", "Age_x", "Age_y"]
        ).agg({"Smaller_id": "count"})
        plot_df_heatmap = plot_df.pivot_table(
            index="Age_x", columns="Age_y", values="Smaller_id"
        ).fillna(0)
        plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
        ax = sns.heatmap(plot_df_heatmap_logged, cmap="jet")
        ax.invert_yaxis()
        ax.set_xlabel("Age (Female)")
        ax.set_ylabel("Age (Female)")
        ax.set_title("(c) #connections per F-F pair", y = -0.15)
        
    def plot_age_relations_heatmap_M_F(self, edges_w_features):
        """Hőtérkép a férfi-nő kapcsolatok arányaira az életkor függvényében"""

        # Férfi-nő kapcsolatok kiszűrése
        edges_w_features_M_F = edges_w_features.loc[
            (edges_w_features["Gender_x"] != edges_w_features["Gender_y"])
        ]

        plot_df = edges_w_features_M_F.groupby(
            ["Gender_x", "Gender_y", "Age_x", "Age_y"]
        ).agg({"Smaller_id": "count"})
        plot_df_heatmap = plot_df.pivot_table(
            index="Age_x", columns="Age_y", values="Smaller_id"
        ).fillna(0)
        plot_df_heatmap_logged = np.log(plot_df_heatmap + 1)
        ax = sns.heatmap(plot_df_heatmap_logged, cmap="jet")
        ax.invert_yaxis()
        ax.set_xlabel("Age (Male)")
        ax.set_ylabel("Age (Female)")
        ax.set_title("(d) #connections per M-F pair", y = -0.15)
        
    def figure5_plot(self, edges_w_features):
        """Figure 5 ábrázolása"""
        plt.figure(figsize=(17.75, 15.6))
        plt.subplot(2, 2, 1)
        self.plot_age_relations_heatmap(edges_w_features)
        plt.subplot(2, 2, 2)
        self.plot_age_relations_heatmap_M_M(edges_w_features)
        plt.subplot(2, 2, 3)
        self.plot_age_relations_heatmap_F_F(edges_w_features)
        plt.subplot(2, 2, 4)
        self.plot_age_relations_heatmap_M_F(edges_w_features)

        
#---------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------------------------------------------------Adatvizualizáció: Figure 6-----------------------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------- 
      

# F_F kapcsolatok
    def pivot_F_F(self, edges_w_features):
        """F-F kapcsolatokhoz pivot tábla létrehozása"""
        edges_w_features_F_F = edges_w_features.loc[
            (edges_w_features["Gender_x"] == 0.0) & (edges_w_features["Gender_y"] == 0.0)]

        df_F_F = edges_w_features_F_F.groupby(
            ["Gender_x", "Gender_y", "Age_x", "Age_y"]).agg({"Smaller_id": "count"})
        df_F_F_pivot = df_F_F.pivot_table(index="Age_x", columns="Age_y", values="Smaller_id").fillna(0)
        
        df_F_F_pivot["F_total"] = df_F_F_pivot.sum(axis=1)

        return df_F_F_pivot

    # F-M kapcsolatok:
    def pivot_F_M(self, edges_w_features):
        """F-M kapcsolatokhoz pivot tábla létrehozása"""
        edges_w_features_F_M = edges_w_features.loc[
            (edges_w_features["Gender_x"] == 0.0) & (edges_w_features["Gender_y"] == 1.0)]

        df_F_M = edges_w_features_F_M.groupby(
            ["Gender_x", "Gender_y", "Age_x", "Age_y"]).agg({"Smaller_id": "count"})
        df_F_M_pivot = df_F_M.pivot_table(index="Age_x", columns="Age_y", values="Smaller_id").fillna(0)
        
        df_F_M_pivot["M_total"] = df_F_M_pivot.sum(axis=1)

        return df_F_M_pivot

    #: M-F kapcsolatok:
    def pivot_M_F(self, edges_w_features):
        """M-F kapcsolatokhoz pivot tábla létrehozása"""
        edges_w_features_M_F = edges_w_features.loc[
            (edges_w_features["Gender_x"] == 1.0) & (edges_w_features["Gender_y"] == 0.0)]

        df_M_F = edges_w_features_M_F.groupby(
            ["Gender_x", "Gender_y", "Age_x", "Age_y"]).agg({"Smaller_id": "count"})
        df_M_F_pivot = df_M_F.pivot_table(index="Age_x", columns="Age_y", values="Smaller_id").fillna(0)
        
        df_M_F_pivot["F_total"] = df_M_F_pivot.sum(axis=1)

        return df_M_F_pivot

    #: M-M kapcsolatok:
    def pivot_M_M(self, edges_w_features):
        """M-M kapcsolatokhoz pivot tábla létrehozása"""
        edges_w_features_M_M = edges_w_features.loc[
            (edges_w_features["Gender_x"] == 1.0) & (edges_w_features["Gender_y"] == 1.0)]

        df_M_M = edges_w_features_M_M.groupby(
            ["Gender_x", "Gender_y", "Age_x", "Age_y"]).agg({"Smaller_id": "count"})
        df_M_M_pivot = df_M_M.pivot_table(index="Age_x", columns="Age_y", values="Smaller_id").fillna(0)
        
        df_M_M_pivot["M_total"] = df_M_M_pivot.sum(axis=1)

        return df_M_M_pivot


    # F_F kapcsolatok aránya
    def pivot_F_F_proportions(self, df_F_F_pivot, df_F_M_pivot):
        """ Az Age x of FEMALE esetében a női kapcsolatok aránya életkor szerint"""
        df_F_F_proportion = df_F_F_pivot
        # Minden női kapcsolatot tartalmazzon
        df_F_F_proportion["T_total"] = df_F_F_pivot["F_total"] + df_F_M_pivot["M_total"]

        df_F_F_proportion = df_F_F_proportion.loc[:, 15.0:50.0].div(
            df_F_F_proportion["T_total"], axis=0)
        return df_F_F_proportion


    # F_M kapcsolatok aránya
    def pivot_F_M_proportions(self, df_F_F_pivot, df_F_M_pivot):
        """ Az Age x of FEMALE esetében a férfi kapcsolatok aránya életkor szerint"""
        df_F_M_proportion = df_F_M_pivot
        # Minden női kapcsolatot tartalmazzon
        df_F_M_proportion["T_total"] = df_F_F_pivot["F_total"] + df_F_M_pivot["M_total"]

        df_F_M_proportion = df_F_M_proportion.loc[:, 15.0:50.0].div(
            df_F_M_proportion["T_total"], axis=0)
        return df_F_M_proportion


    # M_F kapcsolatok aránya
    def pivot_M_F_proportions(self, df_M_F_pivot, df_M_M_pivot):
        """ Az Age x of MALE esetében a női kapcsolatok aránya életkor szerint"""
        df_M_F_proportion = df_M_F_pivot
        # Minden férfi kapcsolatot tartalmazzon
        df_M_F_proportion["T_total"] = df_M_F_pivot["F_total"] + df_M_M_pivot["M_total"]

        df_M_F_proportion = df_M_F_proportion.loc[:, 15.0:50.0].div(
            df_M_F_proportion["T_total"], axis=0
        )
        return df_M_F_proportion


    # M_M kapcsolatok aránya
    def pivot_M_M_proportions(self, df_M_M_pivot, df_M_F_pivot):
        """ Az Age x of MALE esetében a férfi kapcsolatok aránya életkor szerint"""

        df_M_M_proportion = df_M_M_pivot
        # Minden férfi kapcsolatot tartalmazzon
        df_M_M_proportion["T_total"] = df_M_F_pivot["F_total"] + df_M_M_pivot["M_total"]

        df_M_M_proportion = df_M_M_proportion.loc[:, 15.0:50.0].div(
            df_M_M_proportion["T_total"], axis=0
        )
        return df_M_M_proportion


    # F_F kapcsolatok arányai korosztályonként
    def F_F_gen_proportions(self, pivot_F_F_proportions):
        """Női felhasználókra a korosztályonkénti női megoszlások"""

        df_pivot_F_F_proportions = pivot_F_F_proportions

        # Korosztálycsoportok kialakítása
        same_gen = {}
        older_gen = {}
        younger_gen = {}

        for age in df_pivot_F_F_proportions.index:
            same_gen[age] = sum(
                df_pivot_F_F_proportions.loc[age, max(age - 5, 15) : min(age + 5, 50)])

        for age in df_pivot_F_F_proportions.index:
            if age <= 40:
                older_gen[age] = sum(
                    df_pivot_F_F_proportions.loc[age, age + 10 : min(age + 20, 50)])
            else:
                older_gen[age] = 0

        for age in df_pivot_F_F_proportions.index:
            if age >= 25:
                younger_gen[age] = sum(
                    df_pivot_F_F_proportions.loc[age, max(age - 20, 15) : max(age - 10, 15)])
            else:
                younger_gen[age] = 0

        df_pivot_F_F_proportions["F(x-5:x+5)"] = same_gen.values()
        df_pivot_F_F_proportions["F(x+10:x+20)"] = older_gen.values()
        df_pivot_F_F_proportions["F(x-20:x-10)"] = younger_gen.values()
        return df_pivot_F_F_proportions


    # F_M kapcsolatok arányai korosztályonként
    def F_M_gen_proportions(self, pivot_F_M_proportions):
        """Női felhasználókra a korosztályonkénti férfi megoszlások"""
        df_pivot_F_M_proportions = pivot_F_M_proportions

        # Korosztálycsoportok kialakítása
        same_gen = {}
        older_gen = {}
        younger_gen = {}

        for age in df_pivot_F_M_proportions.index:
            same_gen[age] = sum(
                df_pivot_F_M_proportions.loc[age, max(age - 5, 15) : min(age + 5, 50)])

        for age in df_pivot_F_M_proportions.index:
            if age <= 40:
                older_gen[age] = sum(
                    df_pivot_F_M_proportions.loc[age, age + 10 : min(age + 20, 50)])
            else:
                older_gen[age] = 0

        for age in df_pivot_F_M_proportions.index:
            if age >= 25:
                younger_gen[age] = sum(
                    df_pivot_F_M_proportions.loc[age, max(age - 20, 15) : max(age - 10, 15)])
            else:
                younger_gen[age] = 0

        df_pivot_F_M_proportions["M(x-5:x+5)"] = same_gen.values()
        df_pivot_F_M_proportions["M(x+10:x+20)"] = older_gen.values()
        df_pivot_F_M_proportions["M(x-20:x-10)"] = younger_gen.values()
        return df_pivot_F_M_proportions
    
    # M_F kapcsolatok arányai korosztályonként
    def M_F_gen_proportions(self, pivot_M_F_proportion):
        """Férfi felhasználókra a korosztályonkénti női megoszlások"""

        df_pivot_M_F_proportions = pivot_M_F_proportion

        # Korosztálycsoportok kialakítása
        same_gen = {}
        older_gen = {}
        younger_gen = {}

        for age in df_pivot_M_F_proportions.index:
            same_gen[age] = sum(
                df_pivot_M_F_proportions.loc[age, max(age - 5, 15) : min(age + 5, 50)]
            )

        for age in df_pivot_M_F_proportions.index:
            if age <= 40:
                older_gen[age] = sum(
                    df_pivot_M_F_proportions.loc[age, age + 10 : min(age + 20, 50)]
                )
            else:
                older_gen[age] = 0

        for age in df_pivot_M_F_proportions.index:
            if age >= 25:
                younger_gen[age] = sum(
                    df_pivot_M_F_proportions.loc[age, max(age - 20, 15) : max(age - 10, 15)]
                )
            else:
                younger_gen[age] = 0

        df_pivot_M_F_proportions["F(x-5:x+5)"] = same_gen.values()
        df_pivot_M_F_proportions["F(x+10:x+20)"] = older_gen.values()
        df_pivot_M_F_proportions["F(x-20:x-10)"] = younger_gen.values()
        return df_pivot_M_F_proportions

    # M_M kapcsolatok arányai korosztályonként
    def M_M_gen_proportions(self, pivot_M_M_proportions):
        """Férfi felhasználókra a korosztályonkénti férfi megoszlások"""

        df_pivot_M_M_proportions = pivot_M_M_proportions

        # Korosztálycsoportok kialakítása
        same_gen = {}
        older_gen = {}
        younger_gen = {}

        for age in df_pivot_M_M_proportions.index:
            same_gen[age] = sum(
                df_pivot_M_M_proportions.loc[age, max(age - 5, 15) : min(age + 5, 50)]
            )

        for age in df_pivot_M_M_proportions.index:
            if age <= 40:
                older_gen[age] = sum(
                    df_pivot_M_M_proportions.loc[age, age + 10 : min(age + 20, 50)]
                )
            else:
                older_gen[age] = 0

        for age in df_pivot_M_M_proportions.index:
            if age >= 25:
                younger_gen[age] = sum(
                    df_pivot_M_M_proportions.loc[age, max(age - 20, 15) : max(age - 10, 15)]
                )
            else:
                younger_gen[age] = 0

        df_pivot_M_M_proportions["M(x-5:x+5)"] = same_gen.values()
        df_pivot_M_M_proportions["M(x+10:x+20)"] = older_gen.values()
        df_pivot_M_M_proportions["M(x-20:x-10)"] = younger_gen.values()
        return df_pivot_M_M_proportions

    
    def figure6_plot(self, edges_w_features):
        """Figure 6 ábrázolása"""
        plot_input_F_F = self.F_F_gen_proportions(
            self.pivot_F_F_proportions(
                self.pivot_F_F(edges_w_features), self.pivot_F_M(edges_w_features)))

        plot_input_F_M = self.F_M_gen_proportions(
            self.pivot_F_M_proportions(
                self.pivot_F_F(edges_w_features), self.pivot_F_M(edges_w_features)))

        plot_input_M_F = self.M_F_gen_proportions(
            self.pivot_M_F_proportions(
                self.pivot_M_F(edges_w_features), self.pivot_M_M(edges_w_features)))

        plot_input_M_M = self.M_M_gen_proportions(
            self.pivot_M_M_proportions(
                self.pivot_M_M(edges_w_features), self.pivot_M_F(edges_w_features)))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
        sns.lineplot(data=plot_input_F_F, x="Age_x", y="F(x-5:x+5)", color="red", ax=ax1)
        sns.lineplot(data=plot_input_F_F, x="Age_x", y="F(x+10:x+20)", color="green", ax=ax1)
        sns.lineplot(data=plot_input_F_F, x="Age_x", y="F(x-20:x-10)", color="cyan", ax=ax1)
        sns.lineplot(data=plot_input_F_M, x="Age_x", y="M(x-5:x+5)", color="blue", ax=ax1)
        sns.lineplot(data=plot_input_F_M, x="Age_x", y="M(x+10:x+20)", color="deeppink", ax=ax1)
        sns.lineplot(data=plot_input_F_M, x="Age_x", y="M(x-20:x-10)", color="black", ax=ax1)

        ax1.set_ylabel("Proportions")
        ax1.set_xlabel("Age x of Female User")
        ax1.set_title("(a) Proportion of Females friends age", y = -0.15)
        ax1.legend(
            [
                "F(x-5:x+5)",
                "F(x+10:x+20)",
                "F(x-20:x-10)",
                "M(x-5:x+5)",
                "M(x+10:x+20)",
                "M(x-20:x-10)",
            ])

        sns.lineplot(data=plot_input_M_F, x="Age_x", y="F(x-5:x+5)", color="red", ax=ax2)
        sns.lineplot(data=plot_input_M_F, x="Age_x", y="F(x+10:x+20)", color="green", ax=ax2)
        sns.lineplot(data=plot_input_M_F, x="Age_x", y="F(x-20:x-10)", color="cyan", ax=ax2)
        sns.lineplot(data=plot_input_M_M, x="Age_x", y="M(x-5:x+5)", color="blue", ax=ax2)
        sns.lineplot(data=plot_input_M_M, x="Age_x", y="M(x+10:x+20)", color="deeppink", ax=ax2)
        sns.lineplot(data=plot_input_M_M, x="Age_x", y="M(x-20:x-10)", color="black", ax=ax2)

        ax2.set_ylabel("Proportions")
        ax2.set_xlabel("Age x of Male User")
        ax2.set_title("(b) Proportion of Male’s friends’ age", y = -0.15)
        ax2.legend(
            [
                "F(x-5:x+5)",
                "F(x+10:x+20)",
                "F(x-20:x-10)",
                "M(x-5:x+5)",
                "M(x+10:x+20)",
                "M(x-20:x-10)",
            ])       
