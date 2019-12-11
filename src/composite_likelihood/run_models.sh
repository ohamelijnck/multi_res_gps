cd /Users/ohamelijnck/Documents/paper_submissions/nips_camera_ready/src/models/mr_gprn
pip3 install --ignore-installed .
cd /Users/ohamelijnck/Documents/paper_submissions/nips_camera_ready/src/experiments/composite_likelihood
python setup_data.py
cd models
#python m_single_gp.py
python m_cmgp.py 
#python m_cmgp.py 1
