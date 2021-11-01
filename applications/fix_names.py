import os
import glob

files_to_rename = glob.glob(
    '/Users/rbiessel/Documents/InSAR/Toolik/Fringe/evd/**/*')

refdate = '20200525'
for file in files_to_rename:
    # date = os.path.basename(file)
    # date = date[9:17]
    # base = os.path.basename(file).replace(
    #     f'{refdate}-{date}', f'{refdate}-{date}_')
    # new_path = file.split('/')
    # new_path[-1] = base
    # new_path = '/'.join(new_path)

    os.rename(file, file.replace('-', '_'))
