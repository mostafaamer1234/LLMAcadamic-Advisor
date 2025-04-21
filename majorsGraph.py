# majors_graph.py
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


class MajorGraph:
    def __init__(self, csv_path):
        self.df = pd.read_csv(csv_path)
        self.G = nx.DiGraph()
        self.course_types = {
            "Prescribed Courses": "Prescribed",
            "Additional Courses": "Additional",
            "General Education Requirements": "General Education",
            "Supporting Courses and Related Areas": "Supporting",
            "Special Requirements": "Special",
            "General Option Courses": "General Option",
            "Data Science Option Courses": "Data Science Option",
            "Applied Data Sciences Option Summary": "Applied Data Sciences Option",
            "Computational Data Sciences Option Summary": "Computational Data Sciences Option",
            "Statistical Modeling Data Sciences Option Summary": "Statistical Modeling Data Sciences Option",
            "Traditions of Innovation Option Summary": "Traditions of Innovation Option",
            "Writing and Literature in Context Option Summary": "Writing and Literature in Context Option",
            "Accounting Option Summary": "Accounting Option",
            "Business Analytics Option Summary": "Business Analytics Option",
            "Entrepreneurship Option Summary": "Entrepreneurship Option",
            "Financial Services Option Summary": "Financial Services Option",
            "Health Services Option Summary": "Health Services Option",
            "Individualized Business Option Summary": "Individualized Business Option",
            "Management and Marketing Option Summary": "Management and Marketing Option",
            "Ecology Option Summary": "Ecology Option",
            "General Biology Option Summary": "General Biology Option",
            "Genetics & Dev Bio Option Summary": "Genetics & Dev Bio Option",
            "Neuroscience Option Summary": "Neuroscience Option",
            "Plant Biology Option Summary": "Plant Biology Option",
            "Vertebrate Physiology Option Summary": "Vertebrate Physiology Option",
            "Application Development Option Summary": "Application Development Option",
            "Business Applications Option Summary": "Business Applications Option",
            "Cybersecurity Option Summary": "Cybersecurity Option",
            "Networking Option Summary": "Networking Option",
            "Security and Risk Analysis Option Summary": "Security and Risk Analysis Option",
            "Specialized Technology Option Summary": "Specialized Technology Option",
            "Commercial Recreation and Tourism Management Option Summary": "Commercial Recreation Management Option",
            "Community Recreation Management Option Summary": "Community Recreation Option",
            "Outdoor Recreation Management Option Summary": "Outdoor Recreation Option",
            "Professional Golf Management Option Summary": "Professional Golf Option"
        }

        self._build_graph()

    def _extract_courses(self, text):
        if pd.isna(text) or not isinstance(text, str):
            return []
        return [c.strip() for c in text.split(",") if c.strip()]

    def _build_graph(self):
        for _, row in self.df.iterrows():
            major = row["Major Name"]
            self.G.add_node(major, type="major")

            for col, rel in self.course_types.items():
                if col in row:
                    for course in self._extract_courses(row[col]):
                        self.G.add_node(course, type="course")
                        self.G.add_edge(major, course, relation=rel)

    def draw_major_graph(self, major_name):
        sub_nodes = {major_name}
        for neighbor in self.G.neighbors(major_name):
            sub_nodes.add(neighbor)

        subG = self.G.subgraph(sub_nodes)
        pos = nx.spring_layout(subG, seed=42)
        edge_labels = nx.get_edge_attributes(subG, 'relation')

        plt.figure(figsize=(14, 10))
        nx.draw(subG, pos, with_labels=True, node_size=2000, node_color='lightblue', font_size=10, font_weight='bold')
        nx.draw_networkx_edge_labels(subG, pos, edge_labels=edge_labels, font_color='red')
        plt.title(f"Course Graph for {major_name}")
        plt.show()

    def print_hierarchy(self, node, level=0, visited=None):
        if visited is None:
            visited = set()
        if node in visited:
            return
        visited.add(node)

        indent = "  " * level
        print(f"{indent}{node}")

        relation_groups = {}
        for succ in self.G.successors(node):
            rel = self.G.edges[node, succ].get("relation", "Other")
            relation_groups.setdefault(rel, []).append(succ)

        for relation, courses in sorted(relation_groups.items()):
            print(f"{indent}  [{relation}]")
            for i, course in enumerate(sorted(courses)):
                branch = "├── " if i < len(courses) - 1 else "└── "
                print(f"{indent}    {branch}{course}")

mg = MajorGraph("/Users/mostafa/PycharmProjects/LLM/all_abington_majors_combined(4).csv")

if __name__ == "__main__":
    # Create instance of MajorGraph
    mg = MajorGraph("/Users/mostafa/PycharmProjects/LLM/all_abington_majors_combined(4).csv")
    # Print and draw the hierarchy for a specific major
    mg.print_hierarchy("Computer Science, B.S. (Abington)")
    mg.draw_major_graph("Computer Science, B.S. (Abington)")
