cd /Users/ohamelijnck/Documents/paper_submissions/nips_camera_ready/src/models/mr_gprn
pip3 install --ignore-installed .

cd /Users/ohamelijnck/Documents/paper_submissions/nips_camera_ready/src/models/mr_dgp
pip3 install --ignore-installed .

cd /Users/ohamelijnck/Documents/paper_submissions/nips_camera_ready/src/experiments/aq/analyse

#python m_gp_baseline.py 
#python m_dgp_expert.py 
#python m_gprn_aggr.py 0
#python m_gprn_aggr.py 1

#python m_model.py 0
python m_dgp_expert.py 
#python m_gprn_aggr.py 1
#python m_model.py 1
#python m_model.py 2
