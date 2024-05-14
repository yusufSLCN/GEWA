from matplotlib import pyplot as plt
from plotly.offline import plot
import plotly.graph_objs as go


def plot_network(data):
    num_nodes = data.pos.shape[0]
    Xn=[data.pos[k][0] for k in range(num_nodes)]# x-coordinates of nodes
    Yn=[data.pos[k][0] for k in range(num_nodes)]# y-coordinates
    Zn=[data.pos[k][0] for k in range(num_nodes)]# z-coordinates
    Xe=[]
    Ye=[]
    Ze=[]
    for e in range(data.edge_index.shape[1]):
        start = data.edge_index[0][e]
        end = data.edge_index[1][e]
        Xe+=[data.pos[start][0], data.pos[end][0], None]
        Ye+=[data.pos[start][1], data.pos[end][1], None]
        Ze+=[data.pos[start][2], data.pos[end][2], None]
    trace1=go.Scatter3d(x=Xe,
                       y=Ye,
                       z=Ze,
                       mode='lines',
                       line=dict(color='rgb(125,125,125)', width=1),
                       hoverinfo='none'
                       )
    trace2=go.Scatter3d(x=Xn,
                y=Yn,
                z=Zn,
                mode='markers',
                name='actors',
                marker=dict(symbol='circle',
                                size=6,
                                colorscale='Viridis',
                                line=dict(color='rgb(50,50,50)', width=0.5)
                                ),
                hoverinfo='text'
                )
    axis=dict(showbackground=False,
            showline=False,
            zeroline=False,
            showgrid=False,
            showticklabels=False,
            title=''
            )
    layout = go.Layout(
            title="Network graph of the point cloud",
            width=1000,
            height=1000,
            showlegend=False,
            scene=dict(
                xaxis=dict(axis),
                yaxis=dict(axis),
                zaxis=dict(axis),
            ),
        margin=dict(
            t=100
        ),
        hovermode='closest',
        )
    
    data=[trace1, trace2]
    fig=go.Figure(data=data, layout=layout)

    plot(fig)




def plot_point_cloud(data):
    # Plot the point cloud
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data.pos[:, 0], data.pos[:, 1], data.pos[:, 2])
    plt.show()

