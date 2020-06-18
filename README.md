# covid_project
Pipeline which outputs statistics to estimate risks of covid spreading within the camera field of view

# files directory tree
```bash
|-- video_scraping
|   |-- serbia
|   |   |-- serbia_1592484664941.jpg
|   |   |-- serbia_1592484664942.jpg
|   |   |-- ...
|   |
|   |-- himmelried
|   |   |-- himmelried_1592484664941.jpg
|   |   |-- himmelried_1592484664942.jpg
|   |   |-- ...
|
|   `-- 3 -> /proc/15589/fd
|-- fdinfo
|-- net
|   |-- dev_snmp6
|   |-- netfilter
|   |-- rpc
|   |   |-- auth.rpcsec.context
|   |   |-- auth.rpcsec.init
|   |   |-- auth.unix.gid
|   |   |-- auth.unix.ip
|   |   |-- nfs4.idtoname
|   |   |-- nfs4.nametoid
|   |   |-- nfsd.export
|   |   `-- nfsd.fh
|   `-- stat
|-- root -> /
`-- task
    `-- 15589
        |-- attr
        |-- cwd -> /proc
        |-- fd
        | `-- 3 -> /proc/15589/task/15589/fd
        |-- fdinfo
        `-- root -> /

```
