set -x

if [[ ! -f "hfd.sh" ]]; then
    echo "downloading hfd.sh"
    
    wget https://hf-mirror.com/hfd/hfd.sh
    chmod a+x hfd.sh
else
    echo "hfd.sh already exists."
fi
# parquets
bash hfd.sh JasperHaozhe/NL2SQL-Queries --dataset --tool wget
# database
bash hfd.sh JasperHaozhe/NL2SQL-Database --dataset --tool wget

sql_database_path=${sql_database_path:-"sql_data"}
cache_path=${CACHE_PATH:-"/cache"}
mkdir -p $sql_database_path
# parquets
mv NL2SQL-Queries $sql_database_path/
# database
mv NL2SQL-Database $sql_database_path/
cd $sql_database_path/
unzip NL2SQL-Database/sql_database.zip -d $sql_database_path/NL2SQL-Database/
echo "copying $sql_database_path/NL2SQL-Database/* to /cache because utils/sql_executor.py finds databases in /cache by default"
echo "If you want to copy to another folder, set `export CACHE_PATH=` and change the database path in utils/sql_executor.py accordingly"
cp -r $sql_database_path/NL2SQL-Database/* /cache