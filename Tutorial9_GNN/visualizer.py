import pandas
import plotly
from plotly.offline import iplot
import plotly.express as px
import plotly.graph_objects as go


atom_names = {
        1: 'H',
        6: 'C',
        7: 'N',
        8: 'O',
        9: 'F',
        16: 'S',
        17: 'Cl',
    }

atom_colors = {
        'H': 'white',
        'C': 'black',
        'N': 'blue',
        'O': 'red',
        'F': 'green',
        'S': 'yellow',
        'Cl': 'green',
    }

def pyg_to_df(g):

    df = pandas.DataFrame()

    num_nodes = int(g.num_nodes)

    for feat, value in g.items():
        if not hasattr(value, 'detach'):
            continue
        if value.ndim == 0:
            continue
        if value.shape[0] != num_nodes:
            continue
        arr = value.detach().cpu().numpy()
        if arr.ndim == 2:
            for dim in range(arr.shape[1]):
                df[feat + str(dim)] = arr[:, dim]
        else:
            df[feat] = arr

    return df


def draw_plotly(g):

    df = pyg_to_df(g)

    atom_column = None
    if 'attr5' in df.columns:
        atom_column = 'attr5'
    elif 'z' in df.columns:
        atom_column = 'z'
    elif 'z0' in df.columns:
        atom_column = 'z0'
    else:
        raise KeyError("Expected one of 'attr5', 'z', or 'z0' in graph node features")

    atom_numbers = [int(z) for z in df[atom_column]]

    ### Node trace ###
    names  = [atom_names[z] for z in atom_numbers]
    colors = [atom_colors[n] for n in names]

    node_trace=go.Scatter3d(
                x=df['pos0'],
                y=df['pos1'],
                z=df['pos2'],
                mode='markers',
                name='atom',
                marker=dict(symbol='circle',
                                size=[z * 5 for z in atom_numbers],
                                color=colors,
                                line=dict(color='rgb(50,50,50)', width=2)
                                ),
                hovertemplate =
                 '<b>%{text}</b><br>', #+
                 #'<i>(eta,phi,lay)=(%{y:.2f},%{x:.2f},%{z:.2f})</i><br>',
                text = names #['{}<br>E=<i>{:.4f} GeV'.format(cl,en) for cl,en in zip(cell_df['cell_class_label'],cell_df['cell_e'])]
                )
    

    ### Edge trace ###
    src = g.edge_index[0].detach().cpu().numpy()
    dst = g.edge_index[1].detach().cpu().numpy()
    pos = g.pos.detach().cpu().numpy()

    x1list = pos[src, 0]
    x2list = pos[dst, 0]
    y1list = pos[src, 1]
    y2list = pos[dst, 1]
    z1list = pos[src, 2]
    z2list = pos[dst, 2]

    Xe,Ye,Ze = [],[],[]

    for eidx in range(len(x1list)):
        Xe += [x1list[eidx],x2list[eidx],None]
        Ye += [y1list[eidx],y2list[eidx],None]
        Ze += [z1list[eidx],z2list[eidx],None]

    edge_trace = go.Scatter3d(x=Xe,
                y=Ye,
                z=Ze,
                mode='lines',
                line=dict(color='rgb(125,125,125,0.6)', width=3),
                hoverinfo='none'
                )
    

    ### Layout ###
    axis=dict(showbackground=False,
            showline=True,
            zeroline=False,
            showgrid=True,
            showticklabels=True,
            range=[-5,5],
            #title=var
            )

    layout = go.Layout(
            #title="Event "+str(event_idx),
            width=800,
            height=800,
            showlegend=False,
            scene=dict(
                xaxis=axis,
                yaxis=axis,
                zaxis=axis,
                aspectratio=dict(x=1, y=1, z=1),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        #dragmode='pan',
        scene_xaxis_visible=True, scene_yaxis_visible=True, scene_zaxis_visible=True,
        legend=dict(font=dict(size=10),orientation='h'),
        )
    
    ### Plot ###
    data=[node_trace,edge_trace]
    fig=go.Figure(data=data, layout=layout)

    #fig.write_html("display.html")
    iplot(fig)