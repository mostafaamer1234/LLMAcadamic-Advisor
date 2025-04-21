import pandas as pd
import networkx as nx
import re
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

# Robust text cleaning function
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.replace('\xa0', ' ').replace('xa0', ' ').replace('\u00a0', ' ')
    text = re.sub(r'[^A-Za-z0-9\s:,\.()\[\]]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# Extract prerequisites and concurrent with logical relationships
def extract_prereq_and_concurrent(text):
    prereqs, concurrents, concurrent_logic, prereq_logic = [], [], [], []
    if not text:
        return prereqs, concurrents, concurrent_logic, prereq_logic

    text = clean_text(text).upper()

    if 'ENFORCED CONCURRENT AT ENROLLMENT:' in text:
        prereq_text, concurrent_text = text.split('ENFORCED CONCURRENT AT ENROLLMENT:')
    elif 'CONCURRENT COURSES:' in text:
        prereq_text, concurrent_text = text.split('CONCURRENT COURSES:')
    elif 'OR CONCURRENT:' in text or ' OR CONCURRENT ' in text:
        prereq_text, concurrent_text = re.split(r'OR CONCURRENT[: ]', text, maxsplit=1)
    else:
        prereq_text, concurrent_text = text, ""

    # Handle prereq logic (supporting OR)
    prereq_or_groups = re.split(r'\s+OR\s+', prereq_text)
    for group in prereq_or_groups:
        matches = re.findall(r'[A-Z]{2,}\s\d{3}[A-Z]?', group)
        if matches:
            if 'AND' in group:
                prereqs.extend(matches)
                prereq_logic.append(' and '.join(matches))
            elif len(matches) > 1:
                prereqs.extend(matches)
                prereq_logic.append(' or '.join(matches))
            else:
                prereqs.extend(matches)
                prereq_logic.append(matches[0])

    # Handle concurrent logic (supporting OR and AND)
    grouped = re.split(r'\s+OR\s+', concurrent_text)
    for g in grouped:
        if 'AND' in g:
            and_group = re.findall(r'[A-Z]{2,}\s\d{3}[A-Z]?', g)
            if and_group:
                concurrents.extend(and_group)
                concurrent_logic.append(' and '.join(and_group))
        else:
            or_course = re.findall(r'[A-Z]{2,}\s\d{3}[A-Z]?', g)
            if or_course:
                concurrents.extend(or_course)
                concurrent_logic.append(or_course[0])

    prereqs = list(set([p for p in prereqs if p not in concurrents]))
    return prereqs, list(set(concurrents)), concurrent_logic, prereq_logic

# Build course graph accurately from CSV
def build_course_graph(csv_file):
    df = pd.read_csv(csv_file)
    df.fillna("", inplace=True)
    G = nx.DiGraph()

    for _, row in df.iterrows():
        course = clean_text(row["course_number"]).upper()
        prereq_data = clean_text(row["prerequisite_data"])

        G.add_node(course)

        prereqs, concurrents, _, _ = extract_prereq_and_concurrent(prereq_data)

        for prereq in prereqs:
            G.add_edge(prereq, course, type="prereq")

        for concurrent in concurrents:
            G.add_edge(concurrent, course, type="concurrent")

    return G

# Get course prerequisites with formatted logic

def get_course_prerequisites(course_code, G):
    if course_code not in G:
        return f"No course named '{course_code}' found."

    prereqs = [u for u, v, d in G.in_edges(course_code, data=True) if d.get("type") == "prereq"]
    concurrents = [u for u, v, d in G.in_edges(course_code, data=True) if d.get("type") == "concurrent"]

    df = pd.read_csv(csv_file_path)
    logic_line = df[df["course_number"].str.upper().str.strip() == course_code.upper()]["prerequisite_data"].values
    _, _, concurrent_logic, prereq_logic = extract_prereq_and_concurrent(logic_line[0] if len(logic_line) else "")

    result = f"Prerequisites for {course_code}:\n"
    if prereq_logic:
        result += f"- Required before: {', or '.join(prereq_logic)}\n"
    elif prereqs:
        result += f"- Required before: {', '.join(prereqs)}\n"
    if concurrent_logic:
        result += f"- Can be taken concurrently: {', or '.join(concurrent_logic)}\n"
    elif concurrents:
        result += f"- Can be taken concurrently: {', '.join(concurrents)}\n"
    if not prereqs and not concurrents:
        result += "- No prerequisites listed."
    return result

# Visualize the course graph clearly
def visualize_graph(G):
    plt.figure(figsize=(16, 12))
    pos = graphviz_layout(G, prog="dot")

    node_shapes = {node: ("d" if "OR_" in node else "o") for node in G.nodes()}
    node_colors = ["lightgreen" if shape == "d" else "lightblue" for shape in node_shapes.values()]

    for shape in set(node_shapes.values()):
        shaped_nodes = [node for node in G.nodes() if node_shapes[node] == shape]
        nx.draw_networkx_nodes(G, pos, nodelist=shaped_nodes,
                               node_shape=shape, node_size=2500,
                               node_color="lightgreen" if shape == "d" else "lightblue",
                               edgecolors="black")

    prereq_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "prereq"]
    concurrent_edges = [(u, v) for u, v, d in G.edges(data=True) if d["type"] == "concurrent"]

    nx.draw_networkx_edges(G, pos, edgelist=prereq_edges, arrowsize=20, edge_color="black")
    nx.draw_networkx_edges(G, pos, edgelist=concurrent_edges, arrowsize=20,
                           edge_color="red", style="dashed")

    nx.draw_networkx_labels(G, pos, {node: G.nodes[node].get("label", node) for node in G.nodes()},
                            font_size=10, font_weight="bold")

    plt.title("Course Requirements Graph (Solid: Prerequisites, Dashed Red: Concurrent)")
    plt.axis("off")
    plt.show()

# Example usage
csv_file_path = "/Users/mostafa/PycharmProjects/LLM/processed_psu_courses.csv"
course_graph = build_course_graph(csv_file_path)


'''
# Test cases
print(get_course_prerequisites("MATH 414", course_graph))
print(get_course_prerequisites("CMPSC 442", course_graph))
print(get_course_prerequisites("MATH 140", course_graph))
print(get_course_prerequisites("PHYS 212", course_graph))
print(get_course_prerequisites("MATH 141", course_graph))
print(get_course_prerequisites("CMPSC 462", course_graph))

# Visualize graph
# visualize_graph(course_graph)



# Print graph info
print("Number of nodes:", course_graph.number_of_nodes())
print("Number of edges:", course_graph.number_of_edges())
print("Sample edges:", list(course_graph.edges(data=True))[:100])

# Print first 5 dictionaries representing nodes and their edges
graph_dict = {node: list(course_graph.successors(node)) for node in list(course_graph.nodes())[:100]}
print("First 5 dictionaries:", graph_dict)


# Function to print hierarchy recursively
def print_hierarchy(G, node, level=0, visited=None, prefix=""):
    if visited is None:
        visited = set()

    if node in visited:
        return  # Avoid cycles

    visited.add(node)

    indent = "  " * level
    print(f"{indent}{prefix}{node}")

    successors = sorted(G.successors(node))
    for i, succ in enumerate(successors):
        new_prefix = "├── " if i < len(successors) - 1 else "└── "
        print_hierarchy(G, succ, level + 1, visited, new_prefix)

# Find root nodes (no prerequisites)
def find_roots(G):
    return [node for node in G.nodes() if G.in_degree(node) == 0]

roots = find_roots(course_graph)
print("Roots:", roots)

print("Course Prerequisite Hierarchy:")
for root in sorted(roots):
    print_hierarchy(course_graph, root)

# Visualize graph

def visualize_graph(G):
    plt.figure(figsize=(14, 10))
    pos = graphviz_layout(G, prog="dot")

    # Define node colors and shapes
    node_colors = ["lightgreen" if "OR_" in node else "lightblue" for node in G.nodes()]
    node_shapes = {node: "d" if "OR_" in node else "o" for node in G.nodes()}

    for shape in set(node_shapes.values()):
        shaped_nodes = [node for node in G.nodes() if node_shapes[node] == shape]
        node_shape = "d" if shape == "d" else "o"
        nx.draw_networkx_nodes(G, pos, nodelist=shaped_nodes, node_size=3000 if shape == "o" else 2000,
                               node_color="lightblue" if shape == "o" else "lightgreen", node_shape=node_shape,
                               edgecolors="black")

    # Define edges by type
    prereq_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "prereq"]
    concurrent_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("type") == "concurrent"]

    nx.draw_networkx_edges(G, pos, edgelist=prereq_edges, arrowstyle="-|>", arrowsize=15,
                          edge_color="black", connectionstyle="arc3,rad=0.2")
    nx.draw_networkx_edges(G, pos, edgelist=concurrent_edges, style="dashed", edge_color="red",
                          arrowstyle="-|>", arrowsize=15, connectionstyle="arc3,rad=0.2")

    node_labels = {node: G.nodes[node].get("label", node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10, font_weight="bold")

    plt.title("Course Graph: Solid=Prereq, Dashed=Concurrent")
    plt.axis("off")
    plt.show()

visualize_graph(course_graph)


'''