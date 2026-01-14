from graphviz import Digraph

def create_architecture_diagram():
    # Initialize the graph
    # 'LR' sets the direction from Left to Right
    dot = Digraph('DualPhase_Architecture', comment='Bengali Dialect & NER System')
    dot.attr(rankdir='LR', splines='ortho', nodesep='0.6', ranksep='0.8')
    
    # --- Define Styles ---
    # We define attributes for different types of nodes based on your request
    
    # Data Nodes (Blueish)
    data_attr = {
        'shape': 'box', 
        'style': 'filled, rounded', 
        'fillcolor': '#e1f5fe', 
        'color': '#01579b', 
        'penwidth': '2',
        'fontname': 'Helvetica'
    }
    
    # Process Nodes (Purpleish)
    process_attr = {
        'shape': 'note', # note shape looks like a document/process
        'style': 'filled', 
        'fillcolor': '#f3e5f5', 
        'color': '#4a148c', 
        'penwidth': '2',
        'fontname': 'Helvetica'
    }
    
    # Model Nodes (Orangeish)
    model_attr = {
        'shape': 'component', # component shape implies a module/model
        'style': 'filled', 
        'fillcolor': '#fff3e0', 
        'color': '#e65100', 
        'penwidth': '2',
        'fontname': 'Helvetica-Bold'
    }
    
    # Result Nodes (Greenish)
    result_attr = {
        'shape': 'hexagon', 
        'style': 'filled', 
        'fillcolor': '#e8f5e9', 
        'color': '#1b5e20', 
        'penwidth': '2',
        'fontname': 'Helvetica-Bold'
    }

    # --- Constructing the Graph ---

    # 1. Dataset Split (Floating Node)
    # Using 'cylinder' shape for database
    dot.node('Split', 
             label='<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><IMG SRC="path/to/db_icon.png"/></TD></TR><TR><TD>Dataset Split</TD></TR><TR><TD><FONT POINT-SIZE="10">Train 80% / Test 20%</FONT></TD></TR></TABLE>>', 
             shape='cylinder', style='filled', fillcolor='#e1f5fe', color='#01579b')

    # Subgraph 1: Data Prep
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Stage 1: Input Processing', style='dashed', color='grey')
        
        # HTML labels allow us to insert image placeholders easily
        label_raw = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><IMG SRC="PLACEHOLDER_ICON"/></TD></TR><TR><TD>Raw Bengali Data</TD></TR></TABLE>>'
        c.node('Raw', label_raw, **data_attr)
        
        c.node('Pre', 'Preprocessing\n(Tokenization)', **process_attr)
        
        label_emb = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD>Word Embedding</TD></TR><TR><TD><FONT POINT-SIZE="10">FastText 300D</FONT></TD></TR></TABLE>>'
        c.node('Emb', label_emb, **data_attr)
        
        c.edge('Raw', 'Pre')
        c.edge('Pre', 'Emb', label='Encoding')

    # Subgraph 2: Dialect Detection
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Stage 2: Dialect Detection', style='dashed', color='grey')
        
        # Group models to keep them vertically aligned
        c.node('Model1', 'Model A:\nCNN + BiLSTM', **model_attr)
        c.node('Model2', 'Model B:\nBiLSTM + MHA', **model_attr)
        
        c.node('Classifier', 'Global Pooling\n& Classifier', **process_attr)
        c.node('Pred_Dialect', 'Predicted Dialect\nCategory', **result_attr)
        
        # Edges within subgraph
        c.edge('Model1', 'Classifier')
        c.edge('Model2', 'Classifier')
        c.edge('Classifier', 'Pred_Dialect')

    # Subgraph 3: NER
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Stage 3: Dialect-Specific NER', style='dashed', color='grey')
        
        c.node('NER_Model', 'BiLSTM-CRF\nArchitecture', **model_attr)
        c.node('Hidden', 'Hidden States', **process_attr)
        c.node('Emission', 'Emission Scores\n(Linear Trans.)', **process_attr)
        c.node('Global_Decode', 'CRF Layer\n(Global Decoding)', **process_attr)
        
        c.edge('NER_Model', 'Hidden', label='Seq. Process')
        c.edge('Hidden', 'Emission')
        c.edge('Emission', 'Global_Decode')

    # Final Output
    label_out = '<<TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0"><TR><TD><IMG SRC="PLACEHOLDER_ICON"/></TD></TR><TR><TD>Final Output</TD></TR><TR><TD><FONT POINT-SIZE="10">NER Tags (BIO Format)</FONT></TD></TR></TABLE>>'
    dot.node('Output', label_out, **result_attr)

    # --- Connections between Clusters ---
    
    # Split to Raw (Dashed line)
    dot.edge('Split', 'Raw', style='dotted', constraint='false')

    # Embedding to Models
    dot.edge('Emb', 'Model1')
    dot.edge('Emb', 'Model2')
    
    # Embedding to NER (The "Re-embedding" path)
    # constraint='false' helps prevents the graph from getting too tall/wide unnecessarily
    dot.edge('Emb', 'NER_Model', label='Re-Embedding', style='dashed', color='#01579b')

    # Dialect Prediction to NER Selection
    dot.edge('Pred_Dialect', 'NER_Model', label='Selects Specific Model', penwidth='3', color='#d50000')

    # NER to Output
    dot.edge('Global_Decode', 'Output')

    # Render the graph
    # view=True will open the file after generating
    dot.render('architecture_diagram', format='png', view=True)
    print("Diagram generated successfully as 'architecture_diagram.png'")

if __name__ == "__main__":
    create_architecture_diagram()