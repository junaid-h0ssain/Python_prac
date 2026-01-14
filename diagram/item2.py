from graphviz import Digraph

def create_dialect_diagram():
    # Initialize the graph
    # 'LR' sets the direction from Left to Right (Horizontal)
    dot = Digraph('Dialect_Detection_Flow', comment='Dialect Detection Architecture')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.6', ranksep='0.6')
    
    # --- Define Styles (Consistent with previous diagram) ---
    # Data/Input Nodes (Blueish)
    data_attr = {
        'shape': 'box', 
        'style': 'filled, rounded', 
        'fillcolor': '#e1f5fe', 
        'color': '#01579b', 
        'penwidth': '2',
        'fontname': 'Helvetica'
    }
    
    # Processing/Layer Nodes (Purpleish)
    process_attr = {
        'shape': 'note', 
        'style': 'filled', 
        'fillcolor': '#f3e5f5', 
        'color': '#4a148c', 
        'penwidth': '2',
        'fontname': 'Helvetica'
    }
    
    # Deep Learning Model Nodes (Orangeish)
    model_attr = {
        'shape': 'component', 
        'style': 'filled', 
        'fillcolor': '#fff3e0', 
        'color': '#e65100', 
        'penwidth': '2',
        'fontname': 'Helvetica-Bold'
    }
    
    # Result/Output Nodes (Greenish)
    result_attr = {
        'shape': 'hexagon', 
        'style': 'filled', 
        'fillcolor': '#e8f5e9', 
        'color': '#1b5e20', 
        'penwidth': '2',
        'fontname': 'Helvetica-Bold'
    }

    # --- Constructing the Graph ---

    # 1. Input Stage
    with dot.subgraph(name='cluster_input') as c:
        c.attr(label='Preprocessing Phase', style='dashed', color='grey')
        
        c.node('Input', 'Input Sentence', **data_attr)
        c.node('Token', 'Tokenization &\nVocabulary Mapping', **process_attr)
        
        # Using HTML label for FastText to include details
        label_emb = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><B>Word Embedding</B></TD></TR><TR><TD><FONT POINT-SIZE="10">FastText (300D)</FONT></TD></TR></TABLE>>'
        c.node('Emb', label_emb, **data_attr)
        
        c.edge('Input', 'Token')
        c.edge('Token', 'Emb', label='Token IDs')

    # 2. Parallel Model Architecture
    with dot.subgraph(name='cluster_models') as c:
        c.attr(label='Feature Extraction Architectures', style='dashed', color='grey')
        
        # Branch 1: CNN + BiLSTM
        label_m1 = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><B>Model A</B></TD></TR><TR><TD>CNN + BiLSTM</TD></TR><TR><TD><FONT POINT-SIZE="10">(Local n-gram features)</FONT></TD></TR></TABLE>>'
        c.node('Model_CNN', label_m1, **model_attr)
        
        # Branch 2: BiLSTM + MHA
        label_m2 = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><B>Model B</B></TD></TR><TR><TD>BiLSTM + MHA</TD></TR><TR><TD><FONT POINT-SIZE="10">(Global Context & Attention)</FONT></TD></TR></TABLE>>'
        c.node('Model_MHA', label_m2, **model_attr)

    # 3. Aggregation & Classification
    with dot.subgraph(name='cluster_output') as c:
        c.attr(label='Classification Head', style='dashed', color='grey')
        
        # Pooling Node
        label_pool = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><B>Global Pooling</B></TD></TR><TR><TD>Max & Average</TD></TR><TR><TD><FONT POINT-SIZE="10">Fixed Size Vector (512D)</FONT></TD></TR></TABLE>>'
        c.node('Pool', label_pool, **process_attr)
        
        c.node('Dense', 'Dense Layer', **process_attr)
        c.node('Softmax', 'Softmax\nActivation', **process_attr)
        
        # Final Output with Icon Placeholder
        label_out = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><IMG SRC="PLACEHOLDER_ICON"/></TD></TR><TR><TD>Dialect Class</TD></TR></TABLE>>'
        c.node('Output', label_out, **result_attr)
        
        c.edge('Pool', 'Dense')
        c.edge('Dense', 'Softmax')
        c.edge('Softmax', 'Output')

    # --- Connections (Wiring the Graph) ---
    
    # Connect Embedding to BOTH models (Parallel split)
    dot.edge('Emb', 'Model_CNN')
    dot.edge('Emb', 'Model_MHA')
    
    # Connect BOTH models to Pooling (Merge/Concat)
    dot.edge('Model_CNN', 'Pool')
    dot.edge('Model_MHA', 'Pool')

    # Render
    dot.render('dialect_detection_diagram', format='png', view=True)
    print("Diagram generated successfully as 'dialect_detection_diagram.png'")

if __name__ == "__main__":
    create_dialect_diagram()