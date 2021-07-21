#!/usr/bin/env python
# coding: utf-8

# # Data preparation
# 
# * Read the required bson files (machine, devices, issue, tickets)
# * Load their file data into pandas dataframes
# * Extract DBRefs ids (use as needed)
# * Document data dimensions
# * Check for data issues (NaNs, blanks)
# * Understand data relationships 
# * Develop ideas on how to detect machine problems and which data could possibly help predict such problems
# 

# In[34]:


import bson
import pandas as pd


# In[35]:


FilesList = ['machine_data', 
             'machine_failure_data',
             'machine_maintaince_cases_data',
             'machine_operational_cases',
             'devices',
             'device_allocation',
             'device_check_in_details',
             'device_compoents',
             'device_map_area',
             'devices_activity_log',
             'devices_operations_log',
             'issue_area',
             'issue_sub_area',
             'tickets']


# In[36]:


# Extract id from DBRefs
def extract_id(x):
    return x.id


# In[37]:


# Read the appropriate file and store it's data in dataframe
def fileToDf(FileName):
    data = bson.decode_file_iter(open(str(FileName + '.bson'), 'rb'))
    data = pd.DataFrame(data)
    return data


# In[38]:


# Build dataframes after reading files
df_mc_data = fileToDf('machine_data')
df_mc_failures = fileToDf('machine_failure_data')
df_mc_maintenance_cases = fileToDf('machine_maintaince_cases_data')
df_mc_operational_cases = fileToDf('machine_operational_cases')
df_devices = fileToDf('devices')
df_device_allocation = fileToDf('device_allocation')
df_device_checkin = fileToDf('device_check_in_details')
df_device_components = fileToDf('device_compoents')
df_device_map_area = fileToDf('device_map_area')
df_devices_activity_log = fileToDf('devices_activity_log')
df_devices_opers_log = fileToDf('devices_operations_log')
df_issue_area = fileToDf('issue_area')
df_issue_sub_area = fileToDf('issue_sub_area')
df_tickets = fileToDf('tickets')


# In[39]:


# Check size of data for machines
mc_dims = []
mc_dims.append(df_mc_data.shape)
mc_dims.append(df_mc_failures.shape)
mc_dims.append(df_mc_maintenance_cases.shape)
mc_dims.append(df_mc_operational_cases.shape)
print(mc_dims)

# Check size of data for devices
devices_dims = []
devices_dims.append(df_devices.shape)
devices_dims.append(df_device_allocation.shape)
devices_dims.append(df_device_checkin.shape)
devices_dims.append(df_device_components.shape)
devices_dims.append(df_device_map_area.shape)
devices_dims.append(df_devices_activity_log.shape)
devices_dims.append(df_devices_opers_log.shape)
print(devices_dims)


# ## Data Interpretation:
# * The 'machine_*' files (tables) log status readings across operations. maintenane and failures
# * machine_data shows there are only TWO physical machines ('PR511' and 'PR504')
# * The 'devices_*' files (tables) are LOGICAL assignments of the actual machines to customers/users and log their actual usage
# * The 'issue_*' files (tables) identify the range of issues and sub_issues that can occur
# * The 'tickets' files log specific reported problems (by users presumably). Only 5 out of 11 tickets have compelte information linking back to the device/machine that was used
# 
# * One can trace back from tickets to issues to devices and finally the machine

# ### DBRefs: Extract and follow the link

# In[40]:


# DBRefs can be used to link across the tables
# They are objectIds that need to be extracted and used to locate related row in the other table
# Showing an example here

print("machine_data BEFORE DBRef extraction")
df_mc_data


# In[42]:


#Extract DBRefs for 'devices' column
temp_df_mc_data = df_mc_data
temp_df_mc_data['devices'] = df_mc_data['devices'].apply(extract_id)


# In[45]:


print("machine_data AFTER DBRef extraction. 'devices' column now has ObjectIds.")
temp_df_mc_data


# In[69]:


# Let's follow the objectId link to another table
devId = temp_df_mc_data['devices'][18040]
print("Looking for device with id: ", devId, "\n\n")

# Now we search for devId in 'devices' table
ld = len(df_devices)

for i in range(0, ld-1):
    if df_devices.iloc[i]['_id'] == devId:
        print("*** FOUND the device: ***\n\n")
        print(df_devices.iloc[i])


# ### Check data for NaN values

# In[68]:


#df_device_allocation.isnull().sum()
#df_device_checkin.isnull().sum()
#df_device_components.isnull().sum()
#df_device_map_area.isnull().sum()
#df_devices_activity_log.isnull().sum()
#df_devices_opers_log.isnull().sum()

#df_issue_area.isnull().sum()
#df_issue_sub_area.isnull().sum()
df_tickets.isnull().sum()


# **** Found NaN values in:
# 1) 'devices' table deployedDeviceDate (1)
# 2) 'tickets' table has several NaN values:
#     _id                0
#     ticketId           6
#     issueArea          6
#     issueSubArea       6
#     description        0
#     status             6
#     machine            6
#     operator           7
#     raisedBy           6
#     serviceEngineer    6
#     createdAt          0
#     updatedAt          0
#     _class             0
#     resolution         5
#     actionStatus       5
#     actionTakenBy      5
#     ticket             5
#     dtype: int64

# In[ ]:





# # Finding trends and predicting issues
# Idea: For the machine data tables, each field has values logged at various times. We can create a list of these values (i.e. readings) as "what is happening" with the machine. Maybe this is the input we use to find trends and predict issues?
# 
# ISSUE? : Is there enough data to create a ML model? With only 5 complete tickets, how can we idetnify the trend(s) that lead to an issue occuring?

# In[ ]:




