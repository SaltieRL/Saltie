import requests


class ServerConverter:
    file_status = {}
    local_file = None

    def __init__(self, uploading, server_ip):
        self.uploading = uploading
        self.server_ip = server_ip

    def maybe_upload_replay(self, fn):
        try:
            self._upload_replay(fn)
        except:
            print('catching all errors to keep the program going')

    def _upload_replay(self, fn):
        if not self.uploading:
            self.add_to_local_files(fn)
        with open(fn, 'rb') as f:
            r = ''
            try:
                r = requests.post(self.server_ip, files={'file': f})
            except ConnectionRefusedError as error:
                print('server is down ', error)
                self.add_to_local_files(fn)
            except ConnectionError as error:
                print('server is down', error)
                self.add_to_local_files(fn)
            except:
                print('server is down, general error')
                self.add_to_local_files(fn)

            try:
                print('Upload', r.json()['status'])
                self.file_status[fn] = True
            except:
                print('error retrieving status')

    def add_to_local_files(self, fn):
        if fn not in self.file_status:
            self.file_status[fn] = False

    def retry_files(self):
        for key in self.file_status:
            if not self.file_status[key]:
                print('retrying file:', key)
                self.maybe_upload_replay(key)
        print('all files retried')
