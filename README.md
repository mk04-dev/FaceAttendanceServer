# FaceAttendanceServer

<details open>
<summary>Install</summary>

```sh
    # Clone the YOLOv5 repository
    git clone https://github.com/mk04-dev/FaceAttendanceServer.git

    # Navigate to the cloned directory
    cd FaceAttendanceServer

    # Create environment
    python -m venv .env

    # Active environment
    ## Windows
    ./.env/Scripts/activate

    ## Linux
    source ./.env/bin/activate

    # Install required packages
    pip install -r requirements.txt
```

</details>
<details open>
<summary>Running</summary>

- Change database connection in database.py
- Create table in obiz db

```sql
CREATE TABLE `person_embedding` (
	`IDX` INT(10) UNSIGNED NOT NULL AUTO_INCREMENT,
	`PARTY_ID` VARCHAR(20) NOT NULL COLLATE 'utf8mb3_general_ci',
	`EMBEDDING` BLOB NOT NULL,
	`CREATED_DATE` DATETIME NULL DEFAULT (now()),
	`UPDATED_DATE` DATETIME NULL DEFAULT (now()),
	PRIMARY KEY (`IDX`) USING BTREE,
	INDEX `FK_person_embedding_person` (`PARTY_ID`) USING BTREE,
	CONSTRAINT `FK_person_embedding_person` FOREIGN KEY (`PARTY_ID`) REFERENCES `person` (`PARTY_ID`) ON UPDATE NO ACTION ON DELETE NO ACTION
)
COMMENT='Embeddings for face recognition'
COLLATE='utf8mb3_general_ci'
ENGINE=InnoDB
;
```

```sh
    # Run Redis server
    ## Windows
    ./Redis/redis-server.exe ./Redis/redis.windows.conf
    ## Linux
    ./Redis/redis-server.exe

    # Run server
    ./.env/Scripts/activate

    python main.py
```

</details>
