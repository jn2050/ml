
# Run button
/users/mluser/anaconda3/pkgs/notebook-6.0.1-py37_0/lib/python3.7/site-packages/notebook/static/style/style.min.css
/users/mluser/anaconda3/pkgs/notebook-6.0.1-py37_0/lib/python3.7/site-packages/notebook/static/style/ipython.min.css

div.run_this_cell {
    display: none;  --> display: block;
    ...
}

# Enable extensions
# The syntax for this is jupyter nbextension enable followed by the path to the desired extension’s main file
# For example, to enable scratchpad, you would type the following: 
sudo -E jupyter nbextension enable scratchpad/main --sys-prefix
jupyter nbextension list


conda install -y -c conda-forge jupyterlab
jupyter labextension install @aquirdturtle/collapsible_heading